# Why PyWhispr stalls when the CPU is busy (investigation, no code changes)

Symptom: transcription runs on the GPU, yet when the machine's CPU is loaded the
app slows down and the desktop hangs briefly.

## What actually runs where

| Stage | Thread | CPU or GPU | Latency-sensitive |
| --- | --- | --- | --- |
| Mic capture (`audio.py`) | PortAudio callback thread | CPU, trivial (`rms_level` on 1600 samples) | yes, but cheap |
| Mel features (`onnx_asr` `OnnxPreprocessor`) | STT worker | GPU when a GPU provider is first in the list (`loader.py:202-206`) | no, one call |
| Encoder (`nemo.py:95`) | STT worker | GPU, one `run()` for the whole utterance | no |
| **TDT/RNNT greedy decode (`asr.py:192-229`)** | STT worker | **Python loop on the CPU driving one tiny `decoder_joint.run()` per frame/token** | **yes — this is the hot loop** |
| Filler/vocab/join passes (`app.py:_cleaned/_corrected/_joined`) | Qt main thread | CPU, pure Python, milliseconds on a sentence | no |
| Ducking (`ducking.py`) | **Qt main thread**, inside `_start_recording`/`_stop_recording` | CPU + COM IPC to the audio service | yes |
| Global keyboard hook (pynput `WH_KEYBOARD_LL`) | pynput listener thread | CPU, Python callback **per keystroke, system-wide** | **yes, hard real-time** |
| Overlay waveform | Qt main thread | CPU, 24-bar Python `_tick` at 60 fps (`waveform.py:FRAME_MS = 16`) | yes (visible stutter) |

## Root cause candidates

### 1. Decoding is a CPU-bound host loop, not GPU work (primary)

`_AsrWithTransducerDecoding._decoding` (`asr.py:192`) is a Python `while` loop:
per encoder frame it calls `self._decode(...)` → one `decoder_joint.run()`, takes
`logits.argmax()`, and steps. For 15 s of speech that is a few hundred
serialised session calls of ~1 ms of maths each. The wall time is dominated by
Python bytecode, numpy scalar work, ORT's per-`run()` setup and the CPU↔GPU
copy/sync at each call — all on one CPU thread. The GPU is idle most of the
time, so a loaded CPU slows transcription almost linearly. The repo already
measured the symptom without naming it: `onnx_backend.py:29-33` — "the graph is
thousands of small ops and thread synchronisation dominates: 4 threads 0.43 s,
8 threads 0.81 s, 16 threads 1.96 s".

### 2. ONNX Runtime intra-op threads spin-wait (amplifier)

Nothing in the code disables spinning, so each of the three sessions
(preprocessor, encoder, decoder_joint) keeps `intra_op_num_threads = 4`
(`onnx_backend.py:DEFAULT_THREADS`, applied at `onnx_backend.py:295-298`)
spinning between the thousands of micro-ops. On an idle box spinning wins
latency; on a contended box the spinners burn quantum, get descheduled mid-op,
and the loop in (1) pays a scheduler round-trip per `run()`. Only
`intra_op_num_threads` is set — `session.intra_op.allow_spinning` and
`inter_op_num_threads` are left at ORT defaults.

### 3. GIL contention makes the system-wide keyboard hook late (the "hang")

`PynputHotkeyListener`/`DoubleTapListener` (`hotkey.py:50`, `hotkey.py:227`) use
pynput's `keyboard.Listener`, which on Windows installs a `WH_KEYBOARD_LL` hook
(`pynput/keyboard/_win32.py:240`, `pynput/_util/win32.py:282`). Every keystroke
**anywhere on the desktop** is delivered to a Python callback in our process,
which must acquire the GIL. The decode loop in (1) holds the GIL for its Python
portions on the worker thread; the 60 fps waveform `_tick` holds it on the main
thread. When the CPU is contended, the hook callback misses Windows'
`LowLevelHooksTimeout` (300 ms default) and Windows drops/delays the event —
felt as the whole machine's typing hitching, not just PyWhispr. This is the best
explanation for "hangs briefly", and it is active permanently, not only while
transcribing.

### 4. Blocking COM enumeration on the UI thread

`app.py:577` and `app.py:591` call `self.ducker.duck()` / `.restore()` inline in
the main-thread handlers. `SessionDucker.duck` calls
`AudioUtilities.GetAllSessions()` (`ducking.py:_windows_sessions`), which
enumerates every audio session over COM and then does two COM calls per session.
That is tens to hundreds of milliseconds when the audio service is busy — paid
on the Qt main thread between the hotkey press and the recorder starting, and
again on stop. Opt-in (`duck_other_audio`), so it only bites users who enabled
it.

### 5. No priority anywhere

`grep -riE "priority|SetPriorityClass|nice"` over `src/` finds nothing. PyWhispr
is a tray app, so it is never the Windows foreground process and gets no
foreground scheduling boost: its STT worker and its keyboard-hook thread compete
at normal priority with whatever the user is compiling. The keyboard hook thread
in particular is a real-time-ish obligation running at default priority.

### 6. Minor / ruled out

- `astype(np.float32)` on the audio (`onnx_backend.py:transcribe`) copies a few
  MB once. Irrelevant.
- Capture cannot lose audio to CPU load in a way the user sees as a hang: the
  callback only copies a block and appends it (`audio.py:callback`), and PortAudio
  runs that thread at high priority. Overruns would be logged
  ("Audio input status").
- Paste path is already non-blocking (`injector.py` uses `QTimer` hops, no sleeps).
- `foreground.py` Win32 monitor lookup is a handful of user32 calls.

## Mitigations, ranked (effort vs payoff)

1. **Turn off ORT spinning** — in `OnnxBackend._load_with`, alongside
   `intra_op_num_threads`:
   `options.add_session_config_entry("session.intra_op.allow_spinning", "0")`.
   Two lines, no behaviour change, directly targets (2). Worth measuring both
   ways under `stress`-style load, since it costs a little on an idle machine.
2. **Raise the process priority class while transcribing** — `ABOVE_NORMAL` via
   `ctypes.windll.kernel32.SetPriorityClass` around the worker submit, or
   permanently for the process. Cheap, targets (1), (3) and (5). Alternative,
   narrower: raise only the pynput listener thread
   (`SetThreadPriority(..., THREAD_PRIORITY_ABOVE_NORMAL)`) so the keyboard hook
   stops being late — that is the fix for the user-visible desktop hitch.
3. **Move ducking off the main thread** — do `duck()`/`restore()` on the STT
   worker or a dedicated thread (COM needs its own apartment init there), or
   simply defer `duck()` with `QTimer.singleShot(0, ...)` after the recorder has
   started. Fixes (4). Note the restore-on-quit path must stay synchronous.
4. **Idle the waveform when nothing is on screen / drop to 30 fps under load** —
   `waveform.py` already stops its timer on `stop()`; the remaining win is
   reducing `FRAME_MS` pressure during `TRANSCRIBING`, when the pill only shows
   text. Small, reduces main-thread GIL pressure during (1).
5. **Cut the decode loop's per-step cost** (bigger, real fix for (1)) — options,
   in increasing effort: pin the `decoder_joint` session to the CPU provider so
   each tiny step skips the GPU launch/sync while the encoder stays on the GPU
   (measure: for a 0.6 B TDT the joint net is small); use ORT IOBinding to keep
   the decoder state on-device; or move to a model export with the greedy loop
   inside the graph. All of these are upstream-of-us `onnx_asr` shapes, so expect
   to fork or contribute rather than configure.
6. **Stop sniffing every keystroke** — the durable fix for (3) is a chord hotkey
   registered with `RegisterHotKey` on Windows (as macOS already does with
   Carbon, `hotkey.py:122`) instead of a global `WH_KEYBOARD_LL` hook. Removes
   the per-keystroke Python callback entirely, but only works for chords;
   double-tap still needs the hook.

## How to confirm before changing anything

- Existing logs already separate the stages: `"Transcription took %.1fs"`
  (`app.py:602`) and `"Transcribed %.1fs of audio in %.2fs"`
  (`onnx_backend.py:transcribe`). Run the same utterance idle vs under a CPU
  load and compare — near-linear degradation confirms (1).
- Time `encode` against `decoding` once (temporary log, or `py-spy dump` on the
  `pywhispr-stt` thread mid-transcription): expect the sample to land inside
  `_decoding`/`_decode`, not in the encoder.
- For (3): `py-spy` the pynput thread, or watch for typing latency while the app
  is otherwise idle — the hook is installed permanently, so the hitch should
  reproduce with no recording in flight.
