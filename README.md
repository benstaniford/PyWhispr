# PyWhispr

Local, hotkey-driven voice dictation — a Superwhisper / Wispr Flow alternative that runs
entirely on your machine with open source models. Press a hotkey, speak while a small
waveform overlay shows you're being heard, press it again, and the transcribed text is
typed into whatever app has focus.

- **100% local** — audio never leaves your machine
- **Fast** — NVIDIA Parakeet TDT 0.6B v3, sub-second transcription of typical dictation
- **Cross-platform** — Apple Silicon Macs (via [MLX](https://github.com/ml-explore/mlx))
  and PCs with NVIDIA GPUs (via ONNX Runtime + CUDA), with CPU fallback
- **Shareable** — an open [HTTP API](#network-api) lets phones, browsers and games on
  your network use the same warm model

## Install

Just downlaod the latest [release]( https://github.com/benstaniford/PyWhispr/releases) for Mac / Windows and run the installer.

On Windows the installer offers to start PyWhispr as soon as it finishes, and registers
it to run automatically when you log on — a tray app is only useful if it's already
there when you reach for the hotkey. To turn the autostart off afterwards, use **Task
Manager → Startup apps**; to skip it at install time, run the MSI with
`msiexec /i PyWhispr-<tag>-windows-x64.msi AUTOSTART=0`. Uninstalling removes it.

### Upgrading

Just run the newer installer. It closes the running copy first — Windows cannot
replace files an app still has open — and offers to start the new one on the last
page. In a silent install (`/qn`) nothing restarts it until you next log on.

On macOS, replace the app in `~/Applications` and launch it; the new copy asks the
old one to exit, so you don't end up with two menu-bar icons. If you ever need to
stop it from a script, `pywhispr quit` (or `PyWhispr.exe quit` on Windows) waits
until it's gone.

## macOS permissions

PyWhispr works with **zero permission grants** out of the box: the global hotkey uses
the Carbon `RegisterEventHotKey` API (the same approach as Superwhisper), which needs
no Input Monitoring, and without Accessibility the transcript is left on the clipboard
with a "press ⌘V" notification instead of being pasted automatically. Microphone
access is prompted automatically on first recording.

For fully automatic pasting, grant one optional permission to the **terminal app you
launch PyWhispr from**, then fully restart that terminal:

- **Accessibility** (System Settings → Privacy & Security) — lets PyWhispr synthesize
  the ⌘V keystroke so text lands in the focused app by itself, with your previous
  clipboard restored afterwards.

You can also start/stop dictation from the tray icon menu — no hotkey needed at all.

**Exception — double-tap hotkeys:** a hotkey like double-tap-Option can't use the
permission-free API; it has to listen for raw modifier taps, which macOS gates behind
**Input Monitoring**. If you pick one via **Change hotkey…**, PyWhispr requests the
permission and shows what to grant; relaunch it afterwards. Chord hotkeys never need
this. (On Windows, double-tap works without any special permission.)

## Windows / NVIDIA notes

- An RTX 50-series GPU (Blackwell) needs `onnxruntime-gpu` ≥ 1.22 (pinned already), an
  NVIDIA driver ≥ 570 and CUDA 12.8+ runtime.
- If CUDA isn't usable, PyWhispr logs a warning and runs on CPU (still fast — Parakeet
  is a 0.6B model). Note that `onnxruntime-gpu` *advertises* `CUDAExecutionProvider`
  whether or not a CUDA runtime is installed, so the real failure only appears when the
  model is loaded; PyWhispr catches that and retries on CPU.
- A failed model load no longer quits the app — it stays in the tray and reports the
  error. See [Logs & troubleshooting](#logs--troubleshooting).

## Settings

Tray menu → **Settings…**. Three tabs:

- **Dictation** — the hotkey, the start-over hotkey, the microphone, the
  auto-stop guard, start/stop sounds, and ducking other applications.
- **Text** — filler removal, continuation joining, the custom vocabulary (and its
  editor), spoken numbers as digits, and the spoken start-over phrases.
- **Advanced** — plugins, GPU acceleration, the network API (or, in the Lite
  build, the transcription server), and buttons for the config and log files.

Everything applies on **Save**, on the next dictation, except the network API and
the plugin switches: the socket is bound and the plugins imported at startup, so
changing those says a restart is needed rather than pretending. The **tray menu**
itself keeps only what you reach mid-sentence — start/stop, recent dictations,
settings, quit.

The microphone list is by device *name*, so unplugging something else does not
quietly move your choice. If the one you picked is not connected, PyWhispr
records with the system default and tells you which and why.

## Configuration

`~/Library/Application Support/PyWhispr/config.toml` (macOS) or
`%APPDATA%\PyWhispr\config.toml` (Windows) — also reachable from **Settings… →
Advanced → Open config file**.

Most of these have a control on the **settings page** (tray menu → **Settings…**);
the rest are typed once and left alone, which is why they are not on it.

| Key | Default | Meaning |
|---|---|---|
| `hotkey` | `<cmd>+<shift>+<space>` / `<ctrl>+<alt>+<space>` | Toggle hotkey — easiest to change via **Settings… → Dictation → Hotkey → Change…**, which records a keypress. Either a chord (modifiers + a letter, digit, `<space>`, arrows/navigation keys or `<f1>`–`<f20>`) or a modifier double-tap like `double-tap:<alt>` |
| `input_device_name` | system default | The microphone, by name — set it from **Settings… → Dictation → Microphone**. Names rather than indices because an index renumbers when any other device is unplugged. Not connected right now? PyWhispr records with the system default and says so |
| `input_device` | system default | Microphone *index* from `pywhispr devices`. Still honoured, and still what an older config uses; `input_device_name` wins where both are set |
| `model_override` | platform default | Any compatible HuggingFace repo id |
| `model_quantization` | none | Windows/Linux only: `"int8"` loads the quantised model — noticeably faster on the CPU, small accuracy cost. See [Speed](#speed) |
| `use_gpu` | `true` | Windows/Linux only: `false` stops GPU acceleration being used without deleting its libraries — **Settings… → Advanced → GPU acceleration → Disable…** sets it. See [Speed](#speed) |
| `max_recording_seconds` | `120` | Auto-stop guard |
| `play_sounds` | `true` | Start/stop audio cues |
| `duck_other_audio` | `false` | Windows only: turn other applications' audio down while recording and put it back when the recording stops |
| `duck_volume` | `0.0` | How loud other applications stay while ducked, as a 0–1 fraction of their current volume — the default silences them; `0.2` keeps them at 20% |
| `paste_delay_ms` | `150` | Clipboard settle time before pasting |
| `clipboard_restore_delay_ms` | `300` | Wait before restoring your old clipboard |
| `join_continuations` | `true` | Add the missing space when you dictate again straight after a previous dictation — see [Continuation joining](#continuation-joining) |
| `lowercase_continuations` | `true` | Also lower-case the new text's first word when the sentence in front of the caret is unfinished |
| `context_chars` | `64` | How many characters before the caret are read to decide the join |
| `context_memory_seconds` | `90` | When the caret can't be read, how long PyWhispr trusts its memory of its own last insertion |
| `remove_fillers` | `true` | Drop *um*, *uh*, *erm* and friends from the transcript — see [Filler removal](#filler-removal) |
| `extra_filler_words` | `[]` | More terms to drop, e.g. `["you know", "hmm", "like"]` (a phrase only matches those words next to each other) |
| `keep_filler_words` | `[]` | Built-in fillers to leave alone, e.g. `["er", "well"]` |
| `vocabulary_enabled` | `true` | Apply your [custom vocabulary](#custom-vocabulary) to transcripts — edit the list via **Settings… → Text → Edit vocabulary…** |
| `vocabulary_fuzzy` | `true` | Also correct a *near* miss on longer terms, not just spacing and capitalisation |
| `numbers_to_digits` | `true` | Write [spoken numbers](#spoken-numbers) as digits — *one one eight zero* becomes `1180` |
| `plugins_enabled` | `true` | Run [plugins](#plugins) — a phrase you say and what happens when you say it |
| `plugin_actions_enabled` | `true` | Let plugins do things as well as rewrite text. `false` keeps their text changes and stops anything reaching outside PyWhispr |
| `plugins` | `{}` | Per-plugin settings, e.g. `[plugins.emoji]` with `enabled = false` |
| `api_enabled` | `true` | Serve the [network API](#network-api) |
| `api_host` | `0.0.0.0` | Bind address — set to `127.0.0.1` to keep it on this machine |
| `api_port` | `9149` | Listening port |
| `api_max_audio_seconds` | `300` | Longest clip accepted per request |
| `api_max_queue` | `4` | Requests in flight before new ones get `503 busy` |

## Recent dictations (recall)

Auto-paste goes wherever the keyboard focus is. Dictate at a text box that quietly
lost focus and the transcript lands somewhere useless — and the audio is gone, so
there is nothing to transcribe again.

PyWhispr keeps the **last 10 transcripts in memory**. Use tray menu → **Recent
dictations…**, pick one from the list, and **Paste** puts it where the cursor is
now. The picker is as tall as it has entries and scrolls past six. Return or a double-click chooses; Escape or the **✕** closes. Each row also
has a **copy button** that puts that transcript on the clipboard and leaves the
picker open. The picker takes the focus while it is open, so the window that had
it is given it back before the paste goes out.

There is no recall hotkey — the tray menu is the only way in, so the feature costs
no keyboard shortcut.

The transcripts are **never written to disk** — they live in memory, capped at ten,
and go when the app does.

## Continuation joining

Each recording is transcribed on its own, so the model capitalises the first word
and adds a full stop every time. Say something in two goes and the halves used to
collide: *"I went to the shop.Then I came home."*

PyWhispr now looks at what is immediately before the cursor and adapts the new
text to it:

- **Missing space** — inserts one, so a full stop is followed by a space.
- **Unfinished sentence** — if the text before the cursor has no `.`, `!` or `?`,
  the new text's first word is lower-cased, so *"I went to the shop"* + *"And then
  came home."* reads as one sentence. Only common joining words (*and*, *but*,
  *which*, *because*, *the*…) are ever lower-cased, so names like *Ben* and
  *March* keep their capital.
- **A full stop is taken as deliberate** — the capital after it is left alone.

**It never edits or deletes text that is already in your document.** At most it
adds one leading space and changes one letter of the text it is about to paste.

On macOS the cursor position is read through the Accessibility API — the same
permission auto-pasting already needs, so nothing extra is requested. Plenty of
apps don't expose it (Electron apps like Slack and VS Code, Java apps, terminals);
there PyWhispr falls back to remembering what it inserted last, forgetting it as
soon as you switch app or after `context_memory_seconds`. On Windows only the
memory fallback is used.

Because this reads a few characters out of whatever window is focused, that text
is **never written to the log** — only its length. Set `join_continuations = false`
to turn the whole thing off, or `lowercase_continuations = false` to keep the
spacing fix without the capitalisation change.

## Filler removal

Speech models are faithful: say *"Um, so I, uh, think so."* and that is exactly
what gets typed. PyWhispr takes the hesitations out, along with the spacing and
punctuation they leave behind:

| You said | You get |
|---|---|
| *Um, so I think so.* | *So I think so.* |
| *I think, uh, we should go.* | *I think we should go.* |
| *I think so, um.* | *I think so.* |
| *Hello. Um. Right then.* | *Hello. Right then.* |
| *Um.* | nothing at all |

The capital moves onto the word that now starts the sentence, a comma pair that
was only holding the filler apart goes with it, a bracket or dash pair is tidied,
and a run of them (*um, uh,*) is removed in one piece. A comma the sentence needs
stays: *"We need milk, um, eggs."* → *"We need milk, eggs."*, and *"If you do, um,
tell me."* keeps the comma that stops it reading as *"if you do tell me"*.

Only a **closed list of hesitation sounds** is removed: *um*, *umm*, *uhm*, *uh*,
*uhh*, *erm*, *er*, *ahem* and their longer spellings. Words that merely look like
filler are left alone, because a wrongly deleted word costs a re-dictation while a
surviving *um* costs a keystroke: *uh-huh* and *uh huh* mean yes, *uh oh* means
trouble, *err on the side of caution* is a verb, and *ER* and *ERM* are an acronym.
Add terms you never mean with `extra_filler_words = ["you know", "hmm"]`, spare a
built-in with `keep_filler_words`, or set `remove_fillers = false`. These are read
at startup, so restart PyWhispr after changing them.

**What this does not do.** Conversational filler — *you know*, *I mean*,
*basically*, *sort of* — is left alone unless you list it yourself. Removing it
reliably needs to understand the sentence: *"you know that I left"* must keep its
words while *"it's, you know, complicated"* should lose them, and no word list
draws that line. (Wispr Flow does this with a fine-tuned LLM cleanup pass in the
cloud; PyWhispr is local and has no such pass.) Note also that recent Parakeet
models drop most hesitations by themselves, so on a good recording this pass
often has nothing to do.


## Spoken numbers

Read a PIN, a phone number, a port or a year out loud and the model writes it in
words. PyWhispr turns a **run of two or more numbers** into digits:

| said | typed |
| --- | --- |
| call me on one one eight zero | `call me on 1180` |
| twenty five people | `25 people` |
| one hundred and eighty pounds | `180 pounds` |
| the port is nine one four nine | `the port is 9149` |
| it was nineteen eighty four | `it was 1984` |
| seven oh two | `702` |

Numbers that compose into one are worked out (*twenty five* → `25`, *two thousand
and twenty four* → `2024`); numbers that cannot are written one after another,
which is what a year or a code sounds like (*twenty twenty* → `2020`). Commas and
dashes between numbers are the model's own — it punctuates where it heard you
pause — so *"One, one, eight, zero."* still comes out `1180.`

**A single number word is always left as a word.** *"I have five apples"* and
*"one of the reasons"* are untouched, which is what keeps the pass out of ordinary
prose. For the same reason a number split up by punctuation needs three or more of
them (a code, not a list), so *"one, two, or three"* is safe.

**What this does not do.** Ordinals (*first*), decimals (*three point five*),
fractions, *double oh seven* and *a hundred* (*a* is not read as one — *a million
thanks* should not become `1000000 thanks`). Set `numbers_to_digits = false`, or
untick **Settings… → Text → Numbers**, to turn the whole thing off.


## Speed

Transcription runs on the CPU unless a CUDA 13 runtime and cuDNN 9 are present.
`onnxruntime-gpu` advertises `CUDAExecutionProvider` whether or not the libraries
are installed and then quietly drops it, so the honest answer comes from
`pywhispr diagnose`, which prints the providers the sessions actually loaded.

Two things make the CPU path fast, both on by default:

- **A thread cap.** onnxruntime uses one thread per core; that is far slower here,
  because Parakeet TDT decodes autoregressively and thread synchronisation
  dominates thousands of small ops. On 15s of speech: 16 threads 1.96s, 8 threads
  0.81s, **4 threads 0.43s**. `stt_threads` overrides it, `0` restores
  onnxruntime's default.
- **Quantisation, when it helps.** `model_quantization` unset means int8 on the
  CPU (~2x faster, small accuracy cost) and full precision on the GPU, where int8
  is ~4x *slower* — the quantised ops have no CUDA kernels. Set it explicitly to
  override. The quantised weights are a separate download.

With a working CUDA runtime the same clip takes **0.12s**. That needs ~1.2GB of
libraries, including a cuFFT build that is only on NVIDIA's own index; PyWhispr
puts the pip CUDA wheels' DLL directories on the search path itself, since nothing
else does.

**Settings… → Advanced → GPU acceleration → Enable…** does the download, and the
same button reads **Disable…** once it is on — that switches it off without
deleting anything, so turning it back on needs no second download.
`pywhispr disable-gpu` is the one that removes the libraries and frees the space.

macOS uses the MLX backend, where none of this applies — and where the row is
therefore not shown at all.


## Custom vocabulary

Product names, colleagues, codenames and jargon come out of any speech model
spelled the way an ordinary English speaker would write them: *beyond trust*,
*pie whisper*, *cubernetes*. **Settings… → Text → Edit vocabulary…** opens a list of terms,
spelled the way you want them written:

```
BeyondTrust
Kubernetes
Endpoint Privilege Management
Jamf

# When the model mishears a word the same way every time, spell out the fix:
pie whisper => PyWhispr
c sharp => C#
```

Two things then happen to every transcript, local or [over the API](#network-api):

- **Spacing and capitalisation are corrected.** *beyond trust*, *Beyond Trust*,
  *beyondtrust* and *beyond-trust* all become *BeyondTrust*. This can't pick the
  wrong word, so it applies to every term in the list.
- **A near miss is corrected** on terms of five characters or more — *cubernetes*
  and *kubernets* both become *Kubernetes*. Short terms are matched exactly only,
  so listing *Jamf* never touches the word *jam*. Set `vocabulary_fuzzy = false`
  to switch this tier off and keep only the first.

The safeguards are deliberately lopsided, because a wrong substitution is worse
than a missed one: the number of words has to agree, so a correction can never
swallow the word next to it; a term plus an ordinary ending (*Kubernetes* →
*kubernetes clusters*) is left alone; common joining words are never rewritten;
and if two terms are equally close to what was said, neither is used.

**What it can't do.** This is a correction pass over the finished transcript,
not a hint to the model — nothing in Parakeet's decoder takes a word list — so a
term the model didn't hear at all can only be recovered with an explicit
`heard => wanted` line. The file lives next to `config.toml` as `vocabulary.txt`
and is never written to the log, only its term count.

## Plugins

A plugin is a phrase you say and something that happens when you say it. One
ships with PyWhispr:

> "That fixed it **thumbs up emoji**" → *That fixed it 👍*

Say **"emote"** instead of "emoji" if you prefer — the two work identically, and a
chain can mix them.

The words in front of the trigger are looked up — the ~200 names people actually use,
then every emoji name Unicode knows — and the whole lot becomes one character.
Chain them and they all convert: *"man emoji gun emoji"* → 👨 🔫.

It also copes with the model getting the words slightly wrong, which it does
constantly, in four tiers that get progressively more forgiving:

| You say | Transcribed as | |
|---|---|---|
| eyeroll | `eyeroll` | spaces are optional, so compounds work |
| thumbs up | `thumb sup` | and so is where the model put them |
| eyeroll | `I roll` | homophones: `I`/`eye`, `hi`/`high`, `won`/`one`, `czech`/`check` |
| party popper | `partly popper` | a near miss, when only one name is that near |

Each tier only sees what the stricter ones above it could not resolve, which is
what keeps the forgiving ones out of trouble: *"I roll"* becomes 🙄 at the
homophone tier, so the fuzzy tier never gets to notice that `iroll` is one letter
from `troll`.

Say something that names no emoji and nothing happens at all: *"send me an
emoji"* is left exactly as you said it. A request also has to either end a clause
or lead a chain, so *"the fire emoji is best"* stays a sentence about an emoji
rather than becoming a request for one.

The punctuation the model invents around it is taken back off: every transcript
arrives with a full stop appended and a comma wherever the model heard your pause,
so *"hello smile emoji"* is transcribed *"Hello, smile emoji."* and would otherwise
paste as *Hello, 🙂.* — you get *Hello 🙂*. That matters beyond looks: Teams and
Slack only render the big version when the message is nothing but emoji. A `!` or
`?` is yours and stays, and a comma doing real work (*"hello, smile emoji, then I
left"*) is left alone.

### Writing one

**Settings… → Advanced → Plugins → Open folder…**, drop in a `.py` file, restart. A plugin is
a module with a `TRIGGERS` list and either or both of two functions:

```python
from pywhispr.plugins.api import Match, Rewrite, Trigger

NAME = "temperature"
TRIGGERS = (Trigger(phrase="temperature", at_segment_end=True),)

def rewrite(match: Match) -> Rewrite | None:
    """Change the text. Say "twenty five degrees temperature" → "25 C"."""
    if not match.words_before:
        return None
    number = match.words_before[-1].text
    if not number.isdigit():
        return None                      # not for us: leave the transcript alone
    return match.claim_from(match.words_before[-1], f"{number} C")

def act(match: Match) -> None:
    """Do something. Runs on its own thread, after the text has been inserted."""
```

`rewrite` says which span of the transcript it claims and what should replace it
— it never returns a whole transcript, because PyWhispr does the splicing. That's
what makes plugins safe to have: text outside a claimed span cannot be disturbed,
a claim is confined to the words the plugin was shown, and an exception, a bad
span or an oversized replacement means that one match is skipped and the
transcript survives untouched. It runs between transcription and paste, so it
must be quick and must not do I/O — that's what `act` is for.

Write `rewrite` as a pure function of its `Match`. It runs on more than one thread
— the UI thread for a dictation, an HTTP request thread for a [network
API](#network-api) transcription, possibly several at once — so it must not touch
any UI and two calls must not tread on each other. `act` has the opposite deal: its
own thread, one at a time, after the text has landed, so it can take its time.

Returning `None` is how a plugin says *those words weren't meant for me*, and it
is the main guard against firing on ordinary speech. A plugin with a `rewrite`
only gets its `act` when that rewrite claimed something.

### Limits worth knowing

- **A plugin can only change the dictation being inserted now**, not text already
  in the document. PyWhispr never sends Backspace to another application, so
  "thumbs up" and "emoji" have to be in the same dictation.
- **Actions never run for [network API](#network-api) requests.** That port has no
  authentication, so anything able to reach it could otherwise trigger local side
  effects by choosing its words. Rewrites do apply there.
- **A plugin is arbitrary Python**, run at every startup with your privileges —
  the same trust as your shell profile. Read what you install. PyWhispr never
  downloads one.
- Under the packaged app a plugin may import the standard library and `pywhispr`;
  third-party packages aren't in the bundle.
- Plugins load at startup. There's no reload, because a plugin that started a
  thread can't be un-imported.

`plugins_enabled = false` turns the lot off; `plugin_actions_enabled = false`
keeps rewrites but stops anything reaching outside PyWhispr; and one at a time:

```toml
[plugins.emoji]
enabled = false
```

## Network API

PyWhispr keeps the Parakeet model loaded the whole time it's running, so it can serve
transcriptions to other devices for free. Phones, browsers, games and scripts on your
network POST audio to `http://<your-machine>:9149` and get text back — no cloud, no API
key, no per-minute billing.

It only ever converts audio to text. It cannot start a recording on the host, read its
microphone, or type anything into it.

> **There is no authentication.** Anything that can reach port 9149 can spend your CPU
> and GPU. That's fine on a home LAN; on a shared or untrusted network set
> `api_host = "127.0.0.1"`, or `api_enabled = false` to close the port entirely.
> Expect a firewall prompt on first launch (macOS "allow incoming connections", Windows
> Defender Firewall) — the API is unreachable from other machines until you allow it.

### `GET /v1/health`

```json
{
  "status": "ready",
  "version": "0.2.4",
  "backend": "parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)",
  "sample_rate": 16000,
  "max_audio_seconds": 300,
  "max_upload_bytes": 38400000,
  "queue_depth": 0,
  "max_queue": 4,
  "pcm_formats": ["f32le", "s16le"]
}
```

`status` is `loading` while the model warms up (roughly the first 10 s, or minutes on a
first run that downloads it), then `ready`. Poll this before sending audio.

### `POST /v1/transcribe`

Three body encodings, whichever suits your client:

| `Content-Type` | Body |
|---|---|
| `audio/wav` | a complete wav file — 16-bit or 32-bit, any sample rate, mono or stereo |
| `application/octet-stream` | headerless PCM, described by the query parameters below |
| `multipart/form-data` | a form field named `audio` |

Query parameters (raw PCM only): `sample_rate` (default `16000`), `channels`
(default `1`), `format` — `f32le` (default) or `s16le`. Downmixing and resampling to
16 kHz mono happen server-side. A wav is recognised by its `RIFF` header whatever
content type you declare.

```json
{
  "text": "hello world",
  "backend": "parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)",
  "audio_seconds": 2.13,
  "processing_seconds": 0.41
}
```

Errors are `{"error": {"code": "...", "message": "..."}}` with a stable `code`:
`model_loading` and `busy` (both 503, with `Retry-After`), `bad_audio` (400),
`unsupported_media_type` (415), `payload_too_large` (413), `transcribe_failed` (500).

Requests share the single STT worker with local dictation, so a request that arrives
mid-dictation simply waits its turn. Beyond `api_max_queue` in flight you get `busy`
rather than an unbounded backlog.

**No codecs.** Wav and raw PCM only — no mp3, m4a or Opus. In particular the browser's
`MediaRecorder` produces WebM/Opus, which won't work; use the Web Audio recipe below.

### Clients

```sh
curl -s http://192.168.1.20:9149/v1/health
curl -s -X POST http://192.168.1.20:9149/v1/transcribe \
     -H 'Content-Type: audio/wav' --data-binary @clip.wav
```

CORS is open, so browsers can call it directly. Send float32 samples straight from an
`AudioContext`:

```js
// buffer: an AudioBuffer captured via getUserMedia + AudioWorklet/ScriptProcessor
const pcm = buffer.getChannelData(0);           // Float32Array
const res = await fetch(
  `http://192.168.1.20:9149/v1/transcribe?sample_rate=${buffer.sampleRate}&format=f32le`,
  {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: pcm},
);
const {text} = await res.json();
```

Unity / C#:

```csharp
var pcm = new byte[clip.samples * 4];
var samples = new float[clip.samples];
clip.GetData(samples, 0);
Buffer.BlockCopy(samples, 0, pcm, 0, pcm.Length);

var url = $"http://192.168.1.20:9149/v1/transcribe?sample_rate={clip.frequency}&format=f32le";
using var req = new UnityWebRequest(url, "POST") {
    uploadHandler = new UploadHandlerRaw(pcm) {contentType = "application/octet-stream"},
    downloadHandler = new DownloadHandlerBuffer(),
};
yield return req.SendWebRequest();
Debug.Log(req.downloadHandler.text);   // {"text": "...", ...}
```

Streaming (partial results over a WebSocket) isn't implemented; the `/v1` prefix leaves
room to add it without breaking these clients.

## Logs & troubleshooting

If the app misbehaves — the overlay says *Loading model…* forever, the hotkey
does nothing, the tray icon disappears — the log has the answer. **Settings… →
Advanced → Open log file**, or find it at:

| | |
|---|---|
| Windows | `%LOCALAPPDATA%\PyWhispr\Logs\pywhispr.log` |
| macOS | `~/Library/Logs/PyWhispr/pywhispr.log` |
| Linux | `~/.local/state/PyWhispr/log/pywhispr.log` |

It rotates at 2 MB (3 backups) and starts with a report of the machine:
versions, ONNX Runtime providers, model cache contents and any proxy/TLS
overrides. Alongside it, `pywhispr-stderr.log` catches raw output from the
packaged builds that have no console — the first-run download progress bar and
native crash tracebacks (`faulthandler`).

For more detail, set `PYWHISPR_DEBUG=1` before starting the app (equivalent to
`--verbose`, but works for the packaged app where there's no command line).
Debug logging records every state transition, so a stuck cycle is obvious.

To reproduce a startup failure with the output in front of you, run the same
checks from a terminal:

```sh
pywhispr diagnose    # environment report, mic check, model load, test transcribe
```

## Corporate proxies / TLS interception

On a network that inspects TLS (Cloudflare Gateway, Zscaler and the like), the
first-run model download would otherwise fail with `CERTIFICATE_VERIFY_FAILED`:
your machine trusts the intercepting CA, but Python verifies against certifi's
bundled list, which can never contain a private CA.

PyWhispr handles this itself. At startup it routes TLS verification through the
operating system's own trust store — Schannel on Windows, SecTrust on macOS,
OpenSSL's configured paths on Linux — so whatever the machine trusts, PyWhispr
trusts. No certificate is bundled or pinned, so this works for any corporate CA
and survives cert rotation. `pywhispr diagnose` reports which CA set is in force:

```
tls verification: system trust store (truststore)
```

That covers the normal case, where the CA is installed machine-wide (it has to
be, or browsers on the machine would fail too). For a CA that is *not* in the
system store, add a bundle as well:

```sh
export SSL_CERT_FILE=/path/to/ca-bundle.pem REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
```

Set both: they cover different HTTP stacks. `SSL_CERT_FILE` is read by Python's
`ssl` module and so covers `httpx`, which is what `huggingface_hub` downloads
with; `REQUESTS_CA_BUNDLE` is honoured only by `requests`. Either way the system
store is still used first, so a bundle holding one corporate CA and none of the
public roots — which is what these variables usually end up pointing at — no
longer breaks downloads.

For `uv` itself, use `export UV_SYSTEM_CERTS=1` — that's a separate Rust/rustls
path which the above doesn't affect.

Once the model is cached, none of this matters: PyWhispr runs offline.

## Building a macOS app bundle

```sh
packaging/make_icns.sh                     # regenerate the .icns from assets/icon.png
uv run --with pyinstaller pyinstaller packaging/pywhispr.spec --noconfirm
cp -R dist/PyWhispr.app ~/Applications/
```

The bundle is menu-bar-only (`LSUIElement`), ad-hoc signed by PyInstaller, and needs no
permissions to run (see above). Because it's built locally it never gets macOS's
quarantine attribute, so Gatekeeper won't object even on managed machines — and the
microphone prompt shows "PyWhispr" instead of your terminal's name.

## Releases

Pushing a tag like `v0.2.0` triggers GitHub Actions to build and attach to the release:

- `PyWhispr-<tag>-macos-arm64.zip` — the `.app` bundle (PyInstaller, ad-hoc signed)
- `PyWhisprLite-<tag>-macos-x86_64.zip` — the Intel-Mac build (macOS 13 Ventura and up).
  It runs no model locally; it sends audio to a full PyWhispr's network API instead, so
  it prompts for that server's address on first launch (change it later from the tray's
  "Set server…"). Installs and configures separately from the full app.
- `PyWhispr-<tag>-windows-x64.msi` — per-user installer (cx_Freeze), no admin required,
  with a launch-on-finish checkbox and an HKCU `Run` entry for autostart

```sh
git tag v0.2.0 && git push origin v0.2.0
```

Bump `version` in `pyproject.toml` and `src/pywhispr/__init__.py` first. Note the
downloaded macOS zip carries the quarantine attribute, so on first launch users must
right-click → Open (the bundle is ad-hoc signed, not notarized).

## Development

```sh
uv run pytest                # unit tests (fast, no model/hardware)
uv run pytest -m model       # downloads and runs the real model on a fixture
uv run python -m pywhispr.ui.overlay   # visual overlay demo
uv run ruff check src tests
```

Architecture: `app.py` runs a small state machine (idle → recording → transcribing →
inserting) on the Qt main thread. The mic callback (PortAudio thread), hotkey listener
(pynput thread) and STT worker all communicate via queued Qt signals. STT backends
implement `stt/base.py::STTBackend` and take in-memory 16 kHz float32 numpy audio, so
they're testable without a microphone. `api.py` is a stdlib `ThreadingHTTPServer` that
funnels every request through that same single-worker executor, so the model is only
ever used by one thread at a time.

## License

GPL-3.0 — see [LICENSE](LICENSE).
