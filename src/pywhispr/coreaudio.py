"""Read and set the output volume of a macOS audio device (CoreAudio, via ctypes).

macOS has no per-application output volume: the CoreAudio process object
(``kAudioProcessClassID``) carries a pid, a bundle id, its devices and three
"is running" flags, every one of them read-only, and ``AVAudioSession``'s
``DuckOthers`` option is ``API_UNAVAILABLE(macos)``. So ducking here has to move
the *device*, and this module is the binding that lets `ducking.py` do it: which
volume controls a device actually has, and how to read and write them.

ctypes rather than pyobjc-framework-CoreAudio, which is not installed (pyobjc
arrives here only as a transitive dependency of pynput) and whose own API notes
disclaim that the framework works correctly from Python. A framework reached
through ``ctypes.util.find_library`` also needs no PyInstaller ``hiddenimports``
entry, because nothing is being bundled — the framework is part of the OS.
`platform_setup.py` already reaches ApplicationServices and CoreGraphics this way.

Nothing here loads a framework at import time, so importing this module on
Windows or Linux is harmless and its tests run everywhere.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import logging
from functools import lru_cache

log = logging.getLogger(__name__)

FRAMEWORK = "CoreAudio"


def _fourcc(code: str) -> int:
    """A CoreAudio four-character selector as the integer it really is."""
    return int.from_bytes(code.encode("ascii"), "big")


SYSTEM_OBJECT = 1  # kAudioObjectSystemObject
DEFAULT_OUTPUT = _fourcc("dOut")  # kAudioHardwarePropertyDefaultOutputDevice
MAIN_VOLUME = _fourcc("vmvc")  # kAudioHardwareServiceDeviceProperty_VirtualMainVolume
CHANNEL_VOLUME = _fourcc("volm")  # kAudioDevicePropertyVolumeScalar
SCOPE_GLOBAL = _fourcc("glob")
SCOPE_OUTPUT = _fourcc("outp")
UNKNOWN_PROPERTY = _fourcc("who?")  # kAudioHardwareUnknownPropertyError
# HDMI is the widest thing anyone plugs in, and channels are only probed at all
# when the device exposes no single control covering its whole output.
MAX_CHANNELS = 8


class _Address(ctypes.Structure):
    """AudioObjectPropertyAddress: what to ask a device about."""

    _fields_ = [
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    ]


def status_text(status: int) -> str:
    """An OSStatus as the four-character code CoreAudio's headers name it by.

    ``'who?'`` is ``kAudioHardwareUnknownPropertyError`` — the answer an
    aggregate device gives when asked about a volume it does not have. Far more
    use in a log than the bare negative integer it arrives as.
    """
    unsigned = status & 0xFFFFFFFF
    raw = unsigned.to_bytes(4, "big")
    if all(0x20 <= byte < 0x7F for byte in raw):
        return f"'{raw.decode('ascii')}' (0x{unsigned:08x})"
    return f"{status} (0x{unsigned:08x})"


@lru_cache(maxsize=1)
def _lib():
    """The CoreAudio framework, or None where there isn't one. Never at import."""
    path = ctypes.util.find_library(FRAMEWORK)
    if path is None:
        return None
    lib = ctypes.CDLL(path)
    address = ctypes.POINTER(_Address)
    # Declared rather than inferred, because two of these are CoreAudio
    # `Boolean` — a single byte. Left undeclared, ctypes reads four and only
    # gets the right answer while the rest of the word happens to be zero.
    lib.AudioObjectHasProperty.restype = ctypes.c_bool
    lib.AudioObjectHasProperty.argtypes = [ctypes.c_uint32, address]
    lib.AudioObjectIsPropertySettable.restype = ctypes.c_int32
    lib.AudioObjectIsPropertySettable.argtypes = [
        ctypes.c_uint32,
        address,
        ctypes.POINTER(ctypes.c_bool),
    ]
    lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
    lib.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32,
        address,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    lib.AudioObjectSetPropertyData.restype = ctypes.c_int32
    lib.AudioObjectSetPropertyData.argtypes = [
        ctypes.c_uint32,
        address,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    return lib


def _framework():
    lib = _lib()
    if lib is None:
        raise OSError(f"Could not load the {FRAMEWORK} framework")
    return lib


def _pointer(value) -> ctypes.c_void_p:
    """A ctypes value as the untyped pointer the property calls take."""
    return ctypes.cast(ctypes.byref(value), ctypes.c_void_p)


def _has_settable(lib, device: int, address: _Address) -> bool:
    """Does this device carry this property, and will it let us write it?

    Both halves matter: a device can report a volume it refuses to change.
    """
    if not lib.AudioObjectHasProperty(device, ctypes.byref(address)):
        return False
    settable = ctypes.c_bool(False)
    status = lib.AudioObjectIsPropertySettable(
        device, ctypes.byref(address), ctypes.byref(settable)
    )
    return status == 0 and bool(settable.value)


@dataclasses.dataclass(frozen=True)
class VolumeControl:
    """One settable scalar volume on one device.

    Held across a duck, so restore puts back the device we actually turned down
    — not whatever happens to be the default output by then.
    """

    device: int  # AudioObjectID
    selector: int  # MAIN_VOLUME, or CHANNEL_VOLUME for a per-channel control
    element: int  # 0 for a device's own control, 1..N per channel
    scope: int = SCOPE_OUTPUT

    def _address(self) -> _Address:
        return _Address(self.selector, self.scope, self.element)

    def get(self) -> float:
        """This control's level, 0.0–1.0."""
        lib = _framework()
        address = self._address()
        value = ctypes.c_float(0.0)
        size = ctypes.c_uint32(ctypes.sizeof(value))
        status = lib.AudioObjectGetPropertyData(
            self.device, ctypes.byref(address), 0, None, ctypes.byref(size), _pointer(value)
        )
        if status != 0:
            raise OSError(f"CoreAudio error {status_text(status)} reading the output volume")
        return value.value

    def set(self, level: float) -> None:
        """Set this control, which the device is free to round.

        Measured: writing 0.16066420 to a USB device reads back 0.16094071, as
        the value goes through the hardware's own volume curve. It costs the dip
        nothing (0.17%, inaudible) and it is why ducking saves the level it
        *read* rather than one it computed — writing that same value back is
        exact, because it was a point the device chose in the first place.
        """
        lib = _framework()
        address = self._address()
        value = ctypes.c_float(level)
        status = lib.AudioObjectSetPropertyData(
            self.device, ctypes.byref(address), 0, None, ctypes.sizeof(value), _pointer(value)
        )
        if status != 0:
            raise OSError(f"CoreAudio error {status_text(status)} setting the output volume")


def default_output_device() -> int | None:
    """Whatever the Mac is playing through now, or None if it has nothing."""
    lib = _framework()
    address = _Address(DEFAULT_OUTPUT, SCOPE_GLOBAL, 0)
    device = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device))
    status = lib.AudioObjectGetPropertyData(
        SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size), _pointer(device)
    )
    if status != 0:
        raise OSError(f"CoreAudio error {status_text(status)} finding the default output")
    return device.value or None  # 0 is kAudioObjectUnknown


def output_volume_controls() -> list[VolumeControl]:
    """Every settable output volume on the default output device, or [].

    An ordered cascade, because devices disagree about where their volume lives.
    'vmvc' is the one selector that covers a device's whole output and keeps its
    channels in balance, and it was present on every real device measured here —
    but an aggregate or Multi-Output device answers 'who?' to all of this and
    simply cannot be ducked, as do HDMI and many external DACs. Hence a list,
    and hence an empty one being an ordinary answer rather than a failure.

    Element 0 and elements 1..N are alternatives, never both: element 0 is the
    device's own control, so scaling it *and* its channels would square the dip.
    """
    lib = _framework()
    device = default_output_device()
    if device is None:
        return []
    for selector in (MAIN_VOLUME, CHANNEL_VOLUME):
        control = VolumeControl(device, selector, 0)
        if _has_settable(lib, device, control._address()):
            return [control]
    channels = (VolumeControl(device, CHANNEL_VOLUME, element) for element in range(1, MAX_CHANNELS + 1))
    return [control for control in channels if _has_settable(lib, device, control._address())]
