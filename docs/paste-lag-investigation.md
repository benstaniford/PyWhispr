# Why the text arrives seconds after the overlay disappears

Symptom, on a busy machine: the "Transcribing…" pill vanishes and the transcript
turns up at the caret a couple of seconds later. PR #13 took transcription from
0.32 s to 0.21 s, which is real but is not this — the pill is hidden *before* any
of the remaining work happens, so the whole visible gap is downstream of the
model.

## What runs in the gap

`app._on_transcribed` (`app.py:615`) hides the overlay at line 626 and then does
the vocabulary pass, the join, and `injector.insert`. So the invisible window is:

| after the overlay hides | where | cost on this machine |
| --- | --- | --- |
| `_corrected` (vocabulary) | Qt main thread | 0 — no `vocabulary.txt` exists, the pass returns immediately |
| `_joined` (continuation) | Qt main thread | 0 — `read_preceding_text` returns `None` at once off macOS (`caret.py:40`), then one `GetForegroundWindow` |
| `clipboard.text()` + `mimeData()` (save the old contents) | Qt main thread | ms, but Qt's read retries with sleeps when the clipboard is held |
| `clipboard.setText(text)` | Qt main thread | **`OleSetClipboard`, delayed render — see below** |
| `paste_delay_ms` hop | QTimer | **150 ms, unconditional** |
| synthesized Ctrl+V | pynput `SendInput` | 5 ms idle, **148 ms median under load** |
| the target app fetching the text | **our** Qt main thread, over COM | **the bottleneck** |
| `clipboard_restore_delay_ms` hop, then restore | QTimer | **300 ms, unconditional** |

Filler removal, the vocabulary and the join are not it, and neither is the
low-level keyboard hook on its own. The two suspects that PR #12 ranked highest
for the *transcription* stall — the host-side decode loop and ORT spinning — are
finished before this window opens.

## The bottleneck: the clipboard write is delayed-render

`QClipboard::setText` on Windows goes through `OleSetClipboard`, which publishes
an `IDataObject` and renders the text **on demand**. The pasting application's
read is therefore a COM call back into the PyWhispr process, served by the Qt
main thread. Whenever that thread is late, the paste is late — by the same
amount, and for a paste that has already been keyed.

`scripts/clipboard_stall_probe.py` isolates it with no keyboard, no focus and no
CPU load: set the clipboard, block the main thread for N ms, and time how long a
separate process waits for its read to complete.

| main thread busy | reader waited | with `OleFlushClipboard` |
| --- | --- | --- |
| 0 ms | 35 ms | 1 ms |
| 200 ms | 210 ms | 1 ms |
| 500 ms | 530 ms | 0 ms |
| 1500 ms | 1516 ms | 0 ms |

One for one, and flushing removes the dependency completely.

Two things follow that were not obvious:

- **The reader holds the clipboard open while it waits.** So a stalled main
  thread does not merely delay one paste; it makes the *next* `OleSetClipboard`
  fail. Provoked deliberately, `setText` then costs 354–402 ms (max 995 ms) and
  returns `CLIPBRD_E_CANT_OPEN` — OLE retries `OpenClipboard` internally and
  gives up. Raw Win32 `OpenClipboard`/`SetClipboardData` on the same machine:
  0.3 ms median, 71 ms max, never fails.
- **`clipboard_restore_delay_ms = 300` is a correctness bug under load, not just
  a delay.** The restore replaces our text after 300 ms, but the target's fetch
  was measured taking up to 1418 ms. Restore first and the paste lands with the
  *previous* clipboard contents, or nothing at all.

## End-to-end numbers

`scripts/lag_probe.py` drives the real `TextInjector` against a paste target in
another process, splitting the wait into `sendinput` (our `SendInput` → the
target's Ctrl+V key event, i.e. the low-level hook chain), `render` (the target's
key event → the text arriving, i.e. the COM call back into us) and the two fixed
QTimer hops. Medians over 8–10 dictations of 360 characters:

| | idle | 24 spinners | 32 spinners + in-process GIL pressure |
| --- | --- | --- | --- |
| clipboard read + set + 150 ms hop | 162 ms | 248 ms | 357 ms (max 1627) |
| `sendinput` (hook chain) | 6.9 ms | 5.1 ms | 148 ms (max 243) |
| `render` (fetch back into us) | 4.1 ms | 33 ms (max 889) | 288 ms (max 1418) |
| **total, keystroke to text** | **178 ms** | **285 ms (max 1426)** | **716 ms (max 2268)** |
| pastes lost entirely | 0 | 0 | **1 of 10** |

The loaded runs also report the main thread being 55–1566 ms late for a 100 ms
timer, which is the same lateness `render` is paying for.

Reading across the columns:

- **`render` is the term that explodes**, and it is the one that depends on us.
- **`sendinput` only matters once the GIL is contended**, not merely when the CPU
  is busy: 24 external spinners left it at 5 ms; adding a pure-Python thread in
  our own process took it to 148 ms. That is the pynput `WH_KEYBOARD_LL`
  callback waiting for the GIL, and it is PR #12's finding (3) confirmed — but it
  is a minority of the gap here.
- **450 ms of the idle total is hard-coded** (`paste_delay_ms` 150 +
  `clipboard_restore_delay_ms` 300) and 150 ms of it is in front of the text
  appearing.

## Recommended fix

1. **Render the clipboard eagerly.** `ctypes.windll.ole32.OleFlushClipboard()`
   immediately after `clipboard.setText(...)` in `TextInjector.insert`, Windows
   only. The pasting app then reads bytes off the clipboard and never waits for
   our event loop; measured 0–1 ms regardless of how starved we are. This is the
   fix for the reported symptom, and it also removes the "next `setText` fails"
   knock-on. One call, no new dependency.
2. **Make the restore safe rather than timed.** With the data flushed, the paste
   no longer depends on us, but restoring after a fixed 300 ms can still put the
   old text back before a starved *target* has pasted. Restore only if our own
   text is still on the clipboard, and give it a longer, adaptive window — a
   wrong paste is worse than a late one, and the audio is gone either way.
3. **Cut `paste_delay_ms`.** Its job was to "let the clipboard settle"; with an
   eager flush there is nothing left to settle. 20–30 ms would take the idle
   end-to-end from ~180 ms to ~50 ms.
4. **Extend PR #13's priority boost across the injection.** `priority.boosted()`
   wraps `transcribe()` only, so it is released at the exact instant this window
   opens. Holding it until `_on_insert_finished` covers the `sendinput` term and
   the QTimer lateness. Cheap, and it targets what is left after (1).

Not recommended: replacing `QClipboard` with raw Win32 `SetClipboardData`. It is
faster and never failed here, but it is a much larger change than one flush call
and it gives up Qt's mime handling for no measured benefit once (1) is in.

## The GPU question

Verified against the installed `onnx_asr` 0.12.0 rather than assumed:

- **Mel/preprocessing is already on the GPU.** `Manager.__init__`
  (`loader.py:203-208`) sets `use_numpy_preprocessors` False whenever the first
  provider is not CPU, and then turns on the Conv-based STFT precisely because
  `op.STFT` has no CUDA kernel and would otherwise bounce to the host
  (`loader.py:158-165`). Nothing to win here.
- **The encoder is already on the GPU**, one `run()` for the whole utterance.
- **What is still CPU-bound and matters is the TDT greedy decode**:
  `_AsrWithTransducerDecoding._decoding` (`asr.py:193-229`) is a Python `while`
  loop doing one tiny `decoder_joint.run()` and one `logits.argmax()` per frame.
  That is the term that degrades near-linearly with CPU load, and moving it into
  the graph (or at least off the host loop) would cut transcription time on a
  busy box.
- **It would not touch this symptom.** Everything measured above happens after
  `transcribe()` has returned: a Win32 clipboard write, a COM round trip, a
  `SendInput`, a keyboard-hook chain and two hard-coded `QTimer` delays. There is
  no arithmetic in it to offload. On this machine transcription is already
  0.2–0.9 s of a 2 s+ gap. The only indirect benefit of more offload is burning
  less CPU ourselves, which leaves the main thread marginally less starved —
  worth far less than one `OleFlushClipboard` call.

## Reproducing

```sh
python scripts/clipboard_stall_probe.py            # focus-free, load-free
python scripts/clipboard_stall_probe.py --flush     # the fix, same conditions

python scripts/lag_probe.py                         # idle baseline
python scripts/lag_probe.py --load 32 --gil         # the reported symptom
python scripts/lag_probe.py --load 32 --gil --flush # the fix, end to end
```

`lag_probe.py` steals the focus (it has to — it sends a real Ctrl+V) and refuses
to paste unless its own target window is in front, so it will not dump text into
whatever you are doing. It kills every process it starts by PID.

In the app itself, `PYWHISPR_TIMING=1` turns on the same stage line
(`perf.py`) plus a report of how late the Qt main thread is:

```
post-transcribe: signal-delivered=+0ms fillers=+1ms overlay-hidden=+2ms
vocabulary=+0ms join=+0ms clipboard-read=+8ms clipboard-set=+16ms
paste-timer=+151ms keystroke-sent=+6ms restore-timer=+302ms
clipboard-restored=+1ms (total 483ms)
```
