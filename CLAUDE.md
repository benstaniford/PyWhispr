# CLAUDE.md

Project notes for working on PyWhispr — a local voice-dictation menu-bar app
(Parakeet STT via MLX on macOS / ONNX on Windows, PySide6 UI).

## Build & test

```sh
uv sync                                   # Python is pinned to 3.12 (see below)
QT_QPA_PLATFORM=offscreen uv run pytest   # headless Qt for CI/agents
uv run pytest -m model                    # opt-in: downloads & runs the real model
uv run ruff check src tests
```

STT backends take in-memory numpy audio, so they're testable without a mic.

## This machine is corporate-managed (Jamf MDM) — expect friction

- **TLS interception breaks downloads.** Use `UV_SYSTEM_CERTS=1` for `uv` (its
  own rustls path — nothing in the app affects it). The app itself handles it:
  `certs.py` calls `truststore.inject_into_ssl()` at startup, so verification
  goes to the OS trust store where the corporate CA already lives. Only reach
  for a hand-built CA bundle if that fails — and note **which var matters
  depends on the stack**: `httpx` (what `huggingface_hub` downloads with) reads
  `SSL_CERT_FILE` and ignores `REQUESTS_CA_BUNDLE`; `requests` is the reverse.
  Injection is unconditional — `certs.py` used to stand aside when either was
  set, and a `SSL_CERT_FILE` naming one corporate `.cer` (no public roots) then
  failed every download the moment `huggingface_hub` 1.x swapped `requests` for
  `httpx`. truststore keeps such a bundle usable as a fallback anyway. Once the
  model is cached, normal runs work offline.
- **Python 3.12**, not newer: `parakeet-mlx` → `librosa`/`numba` need a numpy
  the newest CPython lacks wheels for. `pyproject.toml` caps `<3.14` and pins
  `numba>=0.61` to stop bad backtracking.
- `/Applications` is not writable; install the bundle to `~/Applications`.

## Logging

`pywhispr/logging_setup.py` owns all of it. Everything goes to a rotating file
(`user_log_dir`) because the Windows build is a console-less GUI exe where
stderr is `None` and otherwise nothing survives. It also installs
`sys.excepthook` + `threading.excepthook` (worker-thread crashes used to vanish)
and a Qt message handler. `PYWHISPR_DEBUG=1` turns on debug for the packaged
app; `pywhispr diagnose` reproduces startup in a terminal with output visible.

**Rule:** failures that leave the app unusable must not quit it. A tray app that
exits is indistinguishable from a crash — report the error in the tray tooltip,
the overlay and the log, and stay running.

## Continuation joining (`join.py` + `caret.py`)

Each recording is transcribed alone, so the model capitalises and adds a full
stop every time and consecutive dictations collide. `join.py` is a pure
`(preceding, text) -> str` that adds a space and lower-cases a continuation
word; `caret.py` finds `preceding`.

- **Only ever additive.** `join_text` returns exactly `("" | " ") + text` with at
  most the first letter re-cased — never Backspace, never an edit to text
  already in the document. `app._joined` re-checks that invariant and falls back
  to the raw transcript, because losing a transcript is unrecoverable (the audio
  is gone) and an exception escaping `_on_transcribed` strands the app in
  `INSERTING` where no hotkey works.
- Lower-casing is gated on a **closed list** of function words, not a
  shape heuristic: an unlisted word keeps its capital (today's behaviour),
  whereas a loose rule produces `ben`/`monday`. One-letter words are filtered
  out of the list so `I` is safe structurally.
- **Import `HIServices`, not `ApplicationServices`** — every AX symbol is in the
  former; the umbrella drags in CoreText for nothing. pyobjc is already a
  transitive dep of pynput, so no new dependency and no spec change.
- AX reads need Accessibility (the grant auto-paste already has), so they only
  work **from the bundle** — `AXIsProcessTrusted()` is False for a CLI `python`.
  Set `AXUIElementSetMessagingTimeout`: AX is synchronous IPC on the GUI thread,
  and a wedged Electron app would otherwise freeze the tray.
- Plenty of apps expose no caret (Electron, Java, terminals). That is the
  everyday case, not an error: debug-level only, fall back to memory.
- **Never log the context text** — it is whatever window is focused, possibly a
  password field. Lengths only.
- Testing gotcha: `patch.dict(sys.modules, ...)` restores by wiping the dict, so
  anything imported for real *inside* the patch is deleted on exit. Faking
  `HIServices` while letting `CoreFoundation` import for real tore pyobjc out of
  `sys.modules` and made later tests pass for the wrong reason. Fake both.

## Settings page (`ui/settings_dialog.py`) + the microphone

The tray menu had grown a row per feature. It now carries only what you reach
mid-sentence — start/stop, recent dictations, Settings…, Quit — and every
preference lives on the page. `tray.py` keeps `open_config`/`open_log` because
`open_path` lives there; the settings page calls them.

- **The dialog edits a copy of the config and the app copies fields back**
  (`EDITED_FIELDS` + `app._apply_settings`), never a wholesale swap. The GPU
  buttons on the Advanced tab mutate the *live* config while the window is open
  (`gpu.turn_off` writes `use_gpu` and saves), so replacing the config with the
  window's older copy would silently undo them.
- **Not every config key is on the page.** The model, thread and paste-timing
  knobs stay in `config.toml` — typed once in a lifetime, and a spin box each
  would bury the settings people do change. "Open config file" is on the page.
- **The listener is silenced for the whole visit**, so the nested dialogs (hotkey
  capture, vocabulary editor, GPU) no longer juggle it themselves and cannot
  re-arm it while the settings window is still up.
- `RESTART_FIELDS` is what genuinely cannot change in-process — the API's socket,
  whether plugins were imported — and earns a notification. Everything else is
  rebuilt in `_apply_settings` and applies at the next dictation.
- **The microphone is persisted by name** (`input_device_name`), resolved through
  `audio.find_device` at *every* recording. A PortAudio index is a position in a
  list, so unplugging any *other* device renumbers it and the "chosen" microphone
  silently becomes a different one. The old `input_device` index is still read for
  configs that have one — nothing to migrate, nothing lost on upgrade.
- **A missing device falls back to the default and says so**, once per
  disappearance (`_missing_device`): the whole point of choosing a microphone is
  that the default is the wrong one, so a silent fallback is the bug. The page
  keeps an unplugged choice in the list, marked, or opening settings would reset
  it just by being opened.
- **PortAudio lists every microphone once per host API**, so on Windows three mics
  arrived as ~17 indistinguishable rows plus MME's "Microsoft Sound Mapper - Input"
  and DirectSound's "Primary Sound Capture Driver" (each host API's own
  "the default" pseudo-device — the page already offers that itself). The fix is to
  *show* one host API (`audio.input_devices`), not to de-duplicate by name: two
  genuinely different devices can share a name and name-matching would hide one.
- **That host API is DirectSound, not WASAPI.** WASAPI is the modern one and the
  obvious choice, and it cannot be used: it is shared-mode here and will not
  resample, so opening our 16 kHz stream on it fails outright with
  `Invalid sample rate [PaErrorCode -9997]`, and WDM-KS answers
  `Blocking API not supported yet`. MME records but truncates every name at 31
  characters (`Microphone (Logitech PRO X Wire`). Measure before changing this — a
  list that looks right and cannot record is worse than duplicates.
- **Resolution deliberately sees more than the list does.** `find_device` tries the
  shown devices, then *every* device (`audio.all_input_devices`), then a name MME
  truncated as an unambiguous prefix — a config written before the list was
  narrowed must not strand. Ambiguous prefixes resolve to nothing rather than to a
  coin toss. `display_name` maps such a stored name onto the row that is shown, or
  the page would mark a working microphone "not connected".

## Transcript recall (`history.py` + `ui/history_dialog.py`)

Auto-paste follows the focus, so a dictation aimed at a box that had lost focus
is lost outright — the audio is gone. The last `HISTORY_SIZE` (10) transcripts
are kept **in memory only** and a tray menu entry opens a picker that pastes the
chosen one at the current caret. **No hotkey of its own** — a second global
shortcut is a real cost (it can clash, it needs registering, it needs a config
key and a failure path) for something reached once in a while.

- What is remembered is the **corrected** transcript, *before* `join_text`: the
  join belongs to the caret that dictation was aimed at, and re-pasting later
  lands somewhere else entirely. `_context.invalidate()` for the same reason.
- **The picker steals the focus it is about to paste into.** `remember_foreground()`
  / `restore_foreground()` in `ui/foreground.py` put it back (Win32
  `GetForegroundWindow`/`SetForegroundWindow`; no-ops elsewhere), then the paste
  goes out one `FOCUS_RESTORE_MS` timer hop later — `SetForegroundWindow` only
  works because *we* own the foreground at that moment, having just closed our
  own dialog.
- Per-row copy buttons, not one global one: the row is what identifies the
  transcript, so the button belongs on it. Copying leaves the picker open —
  pasting closes it, copying does not.
- The dictation hotkey is silenced around any modal dialog (`_resume_listeners`),
  so the chord pressed inside one cannot start a recording.
- **Never log the transcripts** — lengths and counts only, like everything else
  that touches the user's words.
- The preview label elides, the copy button does not: a `QLabel`'s minimum width
  is its whole text, so in a row layout the *button* is what gets clipped at the
  dialog's minimum width. `_ElidingLabel` is free to shrink; the full transcript
  is on the tooltip regardless.
- The dialog is sized to its contents (`_fit_list_to_contents`), scrolling past
  `MAX_VISIBLE_ROWS` — otherwise two transcripts sit above a column of nothing
  and ten make a window taller than the screen.
- The copy button's "copied" timer is **bound to the button**
  (`QTimer.singleShot(ms, button, ...)`). Copy, then close the picker inside
  `COPIED_FEEDBACK_MS`, and an unbound timer reaches into a deleted C++ object:
  "libshiboken: Internal C++ object already deleted", app gone.
- Testing gotcha: rows are `setItemWidget` widgets, so `item.text()` is empty —
  assert on the row's `QLabel`, not the item. A scroll range is only real once
  the dialog has been shown.

## Starting over (`reset_hotkey` + `scratch.py`)

Two ways to throw away a sentence that came out wrong:

- **The reset hotkey** (`reset_hotkey`, defaults to the dictation chord with
  Backspace) drops the audio captured so far and keeps the stream open —
  `AudioRecorder.reset()` rebinds `_blocks` rather than clearing it, because the
  PortAudio callback thread may be appending as it runs. Ignored outside
  RECORDING, and a failure there returns rather than stopping: ending the
  recording would insert the very words the user asked to be rid of. Its
  listener is never stopped around a dialog the way the dictation one is — it
  does nothing outside RECORDING, and no dialog can be open while recording.
- **Spoken "clear clear"** (`voice_reset_phrases`). Live keyword detection
  is impossible — transcription only happens once the recording has stopped — so
  this is a pass over the finished transcript that keeps what follows the *last*
  phrase. First of the transcript passes, so fillers, vocabulary and the join all
  work on the surviving words; wrapped by `app._after_reset` like the others,
  with "a suffix of the input" as the tripwire.
  - Telling the command from the same words used as words is the whole problem,
    and both guards are structural rather than heuristic. **The default phrase is
    doubled** ("clear clear"; "scratch scratch" and "reset reset" work the same
    way for anyone who prefers them) — immediate repetition is close to absent
    from natural speech and trivial to say on purpose. **A phrase only counts as
    a segment of its own**: Whisper punctuates, so the command arrives as "Clear
    clear." or ", clear clear," while the innocent use has words glued either
    side. "Please clear that surface" survives both ways.
  - The segment rule applies to whatever is configured, so a user's own
    single-word marker is safe too; an empty list turns the pass off.

## Spoken numbers (`numbers.py`)

Parakeet writes number words as words, so a dictated PIN, phone number, port or
year arrives as a sentence. Another correction pass over the finished transcript,
**after vocab and before plugins**: an entry of the user's may spell a number word
(`s three => S3`, whose key normalises to `sthree` and would no longer match `s 3`),
and a plugin trigger should see the finished digits for the same reason it sees
corrected spellings.

- **Two mechanisms, and the split is the whole design.** *Within* a group words
  compose arithmetically (`twenty five` → 25); *across* groups digit strings are
  concatenated (`one one eight zero` → 1180). Nothing simpler tells those two apart,
  and concatenation gets years right for free — `twenty twenty` → `2020`,
  `nineteen eighty four` → `1984`.
- **`point` is a third thing: a separator, not a value.** It closes the group and
  puts a `.` in the output, so both mechanisms then run either side of it and
  `twenty six point two` and `two six point two` are both `26.2`, `one point two
  five` is `1.25`. It is an ordinary English word, so the guard is the same shape as
  every other one here — structural, not a heuristic: **a number word on each side
  with nothing but spacing between** (`_fraction_follows`), which is what leaves
  `the two point plan` and `one point I want to make` alone. One point per run: a
  version string is not a decimal, so `one point two point three` converts as far as
  `1.2`. Inside the fraction a comma or a **scale word ends the run** rather than
  joining it — `one point five million` is `1.5 million`, and folding the scale in
  would have emitted `1.5000000`, which is the shape this got wrong before decimals
  were handled at all (the fraction was converting *on its own*: `one point 25`).
  `two point oh` is left as words, because the `oh` thresholds below outrank this.
- **`place` and `min_big` are separate on purpose.** `place` is the magnitude of the
  last slot filled, and it is what stops `twenty twenty` composing while `twenty
  five` does: a teen fills tens *and* units, a tens word leaves units open. `hundred`
  multiplies *within* a chunk rather than closing one, so folding it into `place`
  breaks either `two thousand three hundred` or `one hundred thousand`, depending
  which way you fold.
- **A lone number word is never converted.** `MIN_RUN_WORDS = 2` is the
  false-trigger guard and it is structural, like `scratch.py`'s doubled phrase:
  "I have five apples" and "one of the reasons" are prose, and one number word is
  not a sequence.
- **`and` is only ever pending**, absorbed once the next word extends the group and
  otherwise ending the run in front of itself — so a conjunction can never widen a
  span it did not earn. That alone is not enough: `three hundred and four` really
  *is* 304, so `between three hundred and four hundred` parses to `chunk = 304` and
  only the *second* `hundred` reveals the `and` was a conjunction. Hence **a scale
  word that cannot extend backtracks** past the absorbed `and` and ends the run
  there, giving `between 300 and 400`. A bigger lookahead cannot fix this; the
  disambiguator arrives two words late by construction.
- **A run whose first token is a scale word is refused whole**, never partially
  rescanned. `hundred` cannot start a numeral, so `a hundred and ten thousand` would
  otherwise drop its head and emit `a 10000` — much worse than leaving the sentence
  alone. Every other refusal *does* retry from the next token, which is what turns
  `Oh, one hundred.` into `Oh, 100.`
- **A comma or dash always ends a group**, and punctuation is absorbed only
  *between* groups. The model punctuates where it heard a pause and a spoken number
  is all pauses, so absorbing it is what makes `One, one, eight, zero.` → `1180.`
  work at all — but allowing it *inside* a group would turn "I've got twenty, five
  of which are broken" into "I've got 25 of which are broken". The price is
  `twenty-five` no longer converting, which is no loss: it was already written
  correctly.
- **A punctuated run needs 3+ groups, every one a single digit.** A PIN, phone
  number or code is a digit run; a spoken list of two is not. This is what keeps
  `one, two, or three` and `thirty, forty, fifty percent` intact — absorbing
  punctuation without it turns the first of those into `12, or three`.
- **`oh` needs two thresholds.** It is the spoken zero of every phone number *and*
  an interjection: 3+ groups for a run containing one, 4+ single-digit groups for a
  run led by one. Refusing a leading `oh` outright was the first design and it is
  wrong — `oh seven eight one two three four five` is a UK mobile whose leading zero
  is the entire point. `oh` survives filler removal (it is deliberately absent from
  `DEFAULT_FILLERS`, and `PROTECTED_FOLLOWERS` guards it in "uh oh"), so this pass
  is the only thing standing between an interjection and a digit.
- **The tripwire is a proof, not a ratio.** vocab's 0.5–2× length check would reject
  every legitimate conversion (`one one eight zero` → `1180` is 0.22), and a
  `(str, str) -> bool` "skeleton" comparison cannot work: to avoid false-rejecting
  `One, one, eight, zero.` it would have to re-implement the separator rules and the
  `and` flag, and a tripwire that duplicates the pass shares its bugs. So `to_digits`
  returns its **spans** and `is_digit_substitution` checks them — ordered,
  non-overlapping, digits out, nothing but number words in, and re-splicing
  reproduces the text. Whatever the parser does, it can only have replaced a number.
  The decimal point widens both halves of that proof by exactly one step — the output
  may carry one `.` with a digit either side, and `and`/`point` are allowed in the
  span only *between* two number words, never at an edge, so neither can be swallowed
  off the end of a run.
- **The trade taken knowingly:** `twenty twenty` → `2020` and `thirty forty percent`
  → `3040 percent` are the *same shape*. There is no structural rule that keeps one
  and rejects the other, so the config bool is the escape hatch — which is also why
  there is no second knob. Ordinals, fractions, `double oh seven` and `a` as one are
  all out: `a` in particular would turn `a million thanks` into `1000000 thanks`.
- Tokens are **letters only** (`[^\W\d_]+`), not `vocab._WORD`, which folds an
  internal hyphen into one token and matches bare digits. The gap between two words
  must `fullmatch` a separator, which is how a stray digit (`one 2 three`), a
  newline or a full stop keeps its sentence intact.
- **Never log the transcript or the words matched** — span counts only.

## Custom vocabulary (`vocab.py` + `ui/vocab_dialog.py`)

Parakeet takes no word list at decode time (`generate()` gets a mel and nothing
else), so this is a **correction pass over the finished transcript**, applied
before `join_text` — the join decides about the first word, so it has to see the
corrected one. The API path gets it too: vocabulary is a standing preference
about spelling, unlike joining, which needs a caret.

- Storage is `vocabulary.txt` next to `config.toml`, **raw text as the source of
  truth** so the user's comments and ordering survive the editor. Only a
  *leading* `#` is a comment, so `C#` stays usable as a term.
- Matching is on the **normalised** form (case, spaces, hyphens, punctuation all
  stripped) over 1–4 word windows, longest first, exact tier before fuzzy tier.
- The guards all lean the same way — a wrong substitution is worse than a missed
  one, and every one of these came from a test that caught a real
  misbehaviour:
  - `MIN_KEY_LENGTH` — `C#` normalises to `c` and would rewrite every stray
    letter c. Too-short terms are dropped; `c sharp => C#` is the way to say it.
  - **Word counts must match for a fuzzy hit.** `a beyond-trust` is one edit
    from `beyondtrust`, and the `a` is not ours to delete.
  - `MIN_FUZZY_LENGTH` — at four characters, one edit reaches half of English
    (`Jamf`/`jam`).
  - `key.startswith(rule.key)` — `kubernetes clusters` is already right.
  - Ties are dropped, and `CONTINUATION_WORDS` is reused as a
    never-rewrite list. That list is function words, **not a dictionary**, so
    fuzzy is best-effort by design; `vocabulary_fuzzy = false` is the out.
- `app._corrected` wraps it like `app._joined`: the audio is gone, so a bug here
  must degrade to the raw transcript. It can't use the join's "text untouched"
  invariant (corrections rewrite words by definition), so the tripwire is a
  length ratio.
- **Never log the terms** — they are the user's private nouns (colleagues,
  customers, unreleased products) and the log is what they send us. Line numbers
  and counts only.
- Testing gotcha: `PyWhisprApp.__init__` calls `load_vocabulary()`, which reads
  the *developer's real* vocabulary file. The `app` fixture patches it (and
  `create_hotkey_listener`, which was quietly claiming a real global hotkey).

## Rich paste (`richclip.py` + `custom_emoji.py`)

Some things cannot be said in plain text. A Teams custom emoji has no codepoint —
it is a tenant-hosted image behind an `<img itemid=...>` inside a marker element —
so the only way to hand one to another app is HTML on the clipboard.

- **`QMimeData.setHtml()` is unusable on Windows, and the failure is silent.** It
  writes a `CF_HTML` whose header is valid but whose *document* is not: `StartHTML`
  points straight at `<!--StartFragment-->` with no `<html>`/`<body>` root. WebView2
  apps like Teams ignore the whole payload and take the plain-text alternative, which
  looks exactly like "Teams refuses pasted HTML" and is not. Wrap the same fragment
  in a real document and it renders — custom emoji, bold, links, all of it. Hence
  the hand-built header in `richclip.build_cf_html`, and hence `injector._set_rich`
  deliberately not using Qt. **Diagnosing this needs the raw bytes**: enumerate the
  clipboard with `EnumClipboardFormats` and read `"HTML Format"` back, because every
  higher-level view of it looked correct.
- **The transcript stays plain text end to end.** `Rewrite.html` is a *second
  rendering* carried alongside, and `PluginResult.rich` reports where each one landed
  in **output** coordinates. So `join`, `history` and `_api_transcribe` are untouched
  and keep their invariants — the alternative, letting markup into the transcript,
  would have made `_joined`'s "text untouched" check meaningless.
- `Rewrite.text` must therefore still be something the user would accept: the emoji's
  *name*. That is what arrives wherever HTML does not reach, which is the same way
  Teams' own copy degrades, and it is why a failure here costs formatting rather
  than words.
- **The separator goes in both renderings.** A rich span covers the whole
  replacement, so markup that omits the leading space is spliced over it and the
  image arrives glued to the previous word.
- `app._shifted_rich` moves spans by whatever the join prepended (0 or 1). Its
  "text moved unexpectedly" branch is unreachable via `_joined`, whose own tripwire
  fires first — belt-and-braces, and tested directly rather than through the pipeline.
- **Rich spans are GUI-thread state for one dictation cycle**, set only when
  `collect_actions` is true. A network request runs the same pass on its own thread
  and must not touch them; there is a test for exactly that.
- Custom emoji markup can only be **captured**, never derived — there is no
  documented API for listing a tenant's emoji — and **nothing captures it yet**.
  `teams_emoji.extract()` works and is tested; it simply has no caller. Every route
  was rejected for a reason worth keeping: `act` runs after the injector has already
  overwritten the clipboard; `rewrite` must be reentrant and I/O-free and also runs
  on API request threads; a `pywhispr` subcommand puts Teams into the main program's
  command surface, which is what the framework exists to prevent; and a tray entry
  needs plugins to declare menu actions generically, which nobody has asked for.
  So the store is read-only in practice — hand-editing `custom_emoji.json` works.
- The store is JSON rather than a `vocabulary.txt`-style line format because the
  fragments are hundreds of characters of markup; a hand-editable one-per-line file
  would be a fiction, though deleting an entry still works.
- **Nothing in the main program knows what an emoji is.** An earlier version put the
  store in `pywhispr/` and gave `cli.py` a `learn-emoji` subcommand — 19 emoji
  references in the main CLI — which contradicted this file's own claim that the
  framework knows nothing about emoji. `richclip.py` is the exception that proves the
  rule: it stays in the main package because its code is generic (text + spans →
  HTML) and `injector.py` is its real consumer.
- **`emoji` and `teams_emoji` are two plugins, not one importing the other.** They
  cooperate through the mechanism the framework already had: **returning `None`
  declines, and the words go to the next plugin that matched them.** `teams_emoji` is
  asked first, answers for captured names, and stays silent otherwise.
  - **Being asked first is load-bearing.** `emoji`'s fuzzy tier answers almost
    anything — "frown" → "crown", "shipit" → "ship" — so a plugin running after it
    would essentially never get a turn. This was built as a multi-pass engine first,
    with `teams_emoji` at a higher altitude, and it did not work for exactly that
    reason.
  - `Plugin.altitude` is therefore a **priority, not a pipeline stage**: it breaks
    the tie at a shared position in `_candidates`, and nothing runs in sequence. The
    multi-pass version was reverted — ~60 lines of driver plus rich-span remapping,
    buying a capability nothing used. If a plugin ever genuinely needs to consume
    another's *output* (upgrading a Unicode emoji to Teams' own HTML, say, using the
    `itemscope` attribute), that is when to reconsider, and not before.
  - Altitude still earns its place for one concrete reason: load order alone always
    put built-ins first, so a **user's** plugin could never outrank `emoji`.
- **`decorate(text) -> [(start, end, html)]` is the third phase**, and the reason the
  multi-pass idea was not needed. Wanting Teams' own markup for a standard emoji looks
  like "rewrite the character", but it is not a text change at all: the codepoint is
  exactly what should still arrive in Slack, an email or Notepad. Only the *markup*
  differs. So the phase runs after all rewriting, is handed the finished text, and
  returns spans — it **cannot change a character**, which is what makes it safe where a
  second rewriting pass was not: nothing to remap, no invariant to erode, still one
  pass. A malformed or overlapping span is refused, so a decorator cannot fight a
  rewrite or another decorator.
- **`Match.claim_absorbing`** lives in `plugins/api.py` rather than in `emoji.py`,
  because any plugin turning spoken words into a symbol needs it and because it is
  the single place that knows the separator must go into *both* the text and the
  markup — getting that wrong spliced an image over the space and glued it to the
  previous word.
- The stored fragments are **tenant-scoped** (image URLs fetched with the viewer's
  credentials), so the store is not shareable configuration.
- macOS `NSPasteboard` path is written from the documented API and **unverified on a
  real machine**; every failure returns False and falls back to plain text.
- Testing gotcha: app tests never read the developer's real emoji store only because
  `isolated_app` patches `load_plugins` to `[]`, so the plugin modules are never
  imported. A test that loads the real `BUILTINS` *will* read it.

## Plugins (`plugins/` + `plugins/builtin/emoji.py`)

The other passes each do one fixed job; this is the open-ended one — a phrase the
user says and what happens when they say it. Sits **after vocab, before join**: a
trigger should benefit from the user's spellings, and a rewrite can change the
opening word that join then decides about.

- **The framework does the splicing, never the plugin.** A plugin returns a span
  plus a replacement and `engine.apply_plugins` concatenates untouched slices
  around it, exactly like `apply_vocabulary`. That is what makes arbitrary user
  code safe here: text outside a claim cannot be disturbed, and a plugin cannot
  return a whole new transcript, so it cannot lose one. `app._via_plugins` needs
  no length tripwire as a result — the ratio check `_corrected` uses would be the
  wrong instrument anyway, since "thumbs up emoji" legitimately becomes one
  character.
- **A claim is confined to the words the plugin was shown** (`Match.window_start`
  /`window_end`), or a trigger at the end of a long dictation could replace the
  whole thing. The window stretches over punctuation only on a side with *no*
  context word — without that, a command-only plugin ("new paragraph") cannot
  reach the full stop the model put after it and leaves a lone `.` behind.
- **Two phases, split on purpose.** `rewrite` is synchronous inside the pass; `act`
  runs on its own single thread from `_on_insert_finished`. That ordering is the
  point: an action that types, switches window or reads the clipboard has to happen
  *after* the paste. One thread, not a pool, so two plugins cannot fight over the
  keyboard, and never the STT worker, or a slow plugin would delay the next
  dictation.
- **`rewrite` is not GUI-thread-only, so it must be reentrant.** `_api_transcribe`
  runs the pass on its own HTTP request thread — `api.py` is a `ThreadingHTTPServer`
  with `api_max_queue` in flight — so a rewrite can run in several threads at once
  and alongside a local dictation's GUI-thread call. Hence "a pure function of its
  `Match`" in the docs is a requirement, not a style preference: no Qt, no shared
  mutable state. (Caught in review; the docs previously promised the GUI thread.)
- **`_api_transcribe` gets rewrites and never actions.** `api_host` defaults to
  `0.0.0.0` with no authentication, so anything that can reach the port chooses the
  words that arrive; side effects must not be one keyword away from that. Enforced
  by `collect_actions=False` rather than a config flag, which also keeps the pass
  off `self` on a request thread.
- **Returning `None` is the false-trigger guard, and it lives in the plugin** —
  only emoji knows that "send me an emoji" names no emoji. Same problem
  `scratch.py` has, so `at_segment_end` is the same structural answer.
- Emoji is **a curated alias table over a `unicodedata` name index**, no data file
  and no dependency (~3,000 names in ~1.5ms, built lazily). The aliases are not
  decoration: the stdlib carries the *legacy* Unicode 6 names, so "red heart"
  finds nothing (❤ is `HEAVY BLACK HEART`), "smiling face" prefix-matches
  `SMILING FACE WITH HALO`, and the UCD version travels with the Python version.
  **The alias table is consulted before the function-word guard** — "plus one" is
  two function words and would otherwise never be looked up — the same way an
  explicit `heard => wanted` vocabulary line skips the fuzzy tier's guards.
  `like`, `no` and `done` are deliberately *not* aliases: each lands in front of
  the word "emoji" in sentences about emoji.
- **Emoji's position guard is `_is_a_request`, not the Trigger's `at_segment_end`.**
  It has to be: a chain ("man emoji gun emoji") has an ordinary word after its first
  trigger, so the segment rule discarded the first half before the plugin saw it. A
  request is accepted where it ends a clause *or* leads a chain, and a chain link
  only counts when the words between the two triggers name an emoji **in their
  entirety** — that is what separates "man emoji gun emoji" (two requests) from "I
  use the fire emoji and the water emoji", where "and the water" resolves to nothing
  and stays prose. `at_segment_end` remains a framework feature for user plugins.
- **`TRIGGER_WORDS` is "emoji" *and* "emote"**, interchangeably, chains included.
  "emote" is the riskier one — it is an ordinary verb — and it needs no extra guard
  because the two existing ones already cover it: a request must end a clause or lead
  a chain, *and* the words in front must name an emoji. "He began to emote." fails
  both, since "began to" is not an emoji.
- **A trigger word is refused as a name.** The UCD has `EMOJI COMPONENT BALD` and
  three siblings, so the prefix tier answered "emoji emoji" with a hairstyle — and
  "emote" is two edits from "note", so the fuzzy tier answers it with a notepad.
- **Emoji resolves in four tiers, ordered by how much each is guessing**: literal
  (aliases → index exact/prefix/subset → the same with spaces stripped) → `HOMOPHONES`
  → `_fuzzy`. The ordering *is* the safety mechanism, not presentation: a loose tier
  only ever sees what the strict ones could not reach.
  - **`I roll emoji` used to give 🧻.** Not a phonetics failure — `"i roll"` resolved
    to nothing, so the longest-first loop fell back to the single word `"roll"`, which
    prefix-matches `ROLL OF PAPER`. The homophone tier now answers at two words, so
    the one-word fallback is never reached. **That fallback hazard is still live for
    any other failed multi-word phrase** and was left alone deliberately: restricting
    single-word queries would cost real hits like "pizza" (`SLICE OF PIZZA`) and
    "beer" (`BEER MUG`).
  - **Soundex and friends were measured and rejected.** 16 of ~200 alias keys and 105
    of 2,783 index names collide (`cry`/`car`, `smile`/`snail`, `taco`/`taxi`), *and*
    they miss `i`/`eye` and `hi`/`high` outright because they preserve the first
    letter — so they fail on both accuracy and the cases that motivated the work.
    Edit distance alone is worse than useless here: `iroll` is **one** edit from
    `troll` and three from `eyeroll`, so it answers confidently with a troll.
  - **`HOMOPHONES` never reaches the text.** It rewrites a *lookup key*, and the words
    are then replaced by one character, so "eye" cannot land in the user's sentence.
    That is why the same map must not become a general vocabulary pass, where "I"
    would become "eye" in prose. Entries are exact-sound-equal only, must have a
    target that appears in some real emoji name (tested), and pairs where *both*
    spellings name a different emoji are omitted — knight/night, role/roll.
  - `_guarded` is shared by every tier below the alias table, because the fuzzy tier
    without the function-word check answers "the" with 🌳 (two edits from "tree").
  - The alias table is checked **before the length guard** too: "ok" is two characters
    and was silently dead in the table until that ordering.
- **Teams' `NATIVE_IDS` table is hand-maintained, and every entry needs two checks.**
  Nothing about it is derivable: the thumbs up is `yes`, tears-of-joy is `cwl`, newer ones
  look like `1f47f_angryfacewithhorns`. Guessing from `unicodedata` reproduced 59 of 83
  known ids and missed every common one. The ids came out of Teams' own maps in its cached
  web bundles, and **both** checks matter: (1) the id must resolve on the CDN, because
  `face` and `feed` are valid hex so `face_enrollment` and `feed_loaded` look like ids and
  are not — and an id Teams does not recognise makes it **refuse the entire paste,
  silently**; (2) the asset must actually draw that emoji, because a bare id is a
  *reaction* keyed by meaning, so `like` is a face holding a thumb and `laughdog` is a dog
  for a *face* codepoint. No rule separates them, so the exclusions are empirical and
  ~90 reaction ids ship uninspected. There is deliberately **no regenerate script**: it
  would scrape a private cache Microsoft can change, and would still need human eyes.
- **21 aliases exist purely because the legacy names are unreachable by ear**: 🔫 is
  `PISTOL`, 🍔 is `HAMBURGER`, ⛳ is `FLAG IN HOLE`. Verify a codepoint's real
  `unicodedata.name` before adding one — `U+1FA9B` is `SCREWDRIVER`, not a drill,
  and `U+1FA88` is `FLUTE`, not a whistle.
- **Emoji absorbs the punctuation the model invented** (`_claim_span`): the trailing
  full stop, and a comma immediately in front of the name. Both are the model's,
  not the speaker's — every transcript gets a full stop appended, and a comma lands
  wherever it heard the pause before the name, so "hello smile emoji" arrives as
  "Hello, smile emoji." and pasted as-is reads "Hello, 🙂." The full stop also has a
  concrete cost: **Teams and Slack only render the large emoji when the message is
  nothing but emoji.** `!` and `?` stay (the model does not add those), the trailing
  mark goes only at the very end of the transcript so a clause-separating comma
  survives, and the absorbed separator comes back as a single space.
- **`BUILTINS` names modules as strings, imported at runtime** — so the packaged
  builds cannot see them by static analysis and each must be told about the
  subpackage explicitly. The PyInstaller spec `collect_submodules`es
  `pywhispr.plugins.builtin`; cx_Freeze force-includes the whole `pywhispr`
  package (its `packages` list), which sweeps them up incidentally. Miss the
  PyInstaller side and the built-ins vanish from the macOS build alone —
  `ModuleNotFoundError: No module named 'pywhispr.plugins.builtin'`, which is
  exactly how v0.2.16 shipped emoji broken on Mac while working on Windows.
- **No reload.** A plugin that started a thread cannot be un-imported, so the tray
  opens the folder and a restart applies changes. `plugins/__init__.py`
  deliberately re-exports nothing: `api` and `engine` are stdlib-only, and a
  re-export would drag `registry` → `config` → `platformdirs` into every plugin
  that imports a dataclass.
- **Never log the transcript, the matched words or a rewrite** — plugin name,
  spans and counts only, like everything else that touches the user's words.
- Testing gotcha: `isolated_app` patches `load_plugins` as well as
  `load_vocabulary`. Without it the suite loads the *developer's* plugins folder,
  and a plugin of theirs with an `act()` would really run.
- Testing gotcha: **any test that stops a recording must keep mocks off the worker
  thread.** `TestAudioDucking` asserted on its `ducker` mock while the STT worker was
  reaching `backend.transcribe` on another — a latent race that sat harmless for
  months and then hung the suite the moment unrelated work shifted the timing. It
  reports as a hang at whichever test was running, so it looks like a *new* failure
  in innocent code. `_ducked` installs a plain `SimpleNamespace` backend now.
  `TestContinuationJoin` was always safe because `wait_for_worker` drains first.
- Testing gotcha: **do not drive app code from a second thread in `test_app.py`.**
  `unittest.mock` is not thread-safe, so reaching the fixture's mocks off the main
  thread wedges a *later* test whose main thread and STT worker both build child
  mocks — both ended up inside `mock.__getattr__`/`_get_child_mock` and the suite
  hung, deterministically but nowhere near the offending test. Swapping in a
  non-mock backend was not enough; the test had to go. Concurrency belongs in
  `test_plugins.py::TestReentrancy`, which uses the pure engine and no Qt or mocks.
  Diagnosing this needs `PYTHONUNBUFFERED=1` (a `>` redirect block-buffers, so the
  reported test is thousands of lines stale) plus `-o faulthandler_timeout=25` for
  the thread dump.

## GPU acceleration (`gpu.py` + `cuda.py` + `directml.py`)

CUDA and DirectML are alternatives for the same job, so everything outside them
asks `gpu.py`: is a path *possible* here, is one *on*, turn it off.

- The tray entry is gated on **platform, not backend**: an Intel Mac runs the ONNX
  backend too, but its onnxruntime is the CPU build and DirectML has no macOS
  wheel, so the entry could only ever say no. Apple Silicon is on the Metal GPU
  through MLX regardless. `gpu.supported()` is deliberately the same
  `{win32, linux}` list `cuda.can_offer()` and `directml.can_offer()` gate on
  themselves, rather than a second policy that can drift.
- **`QAction.triggered` is `triggered(bool checked = false)`, and PySide6 hands
  that bool to any slot that will accept one.** Connected straight to
  `app._enable_gpu`, that meant every click passed `asked_by_user=False` — which
  skipped the "not available" answer and instead offered macOS users a 1.2 GB CUDA
  download of *manylinux* wheels (`cuda.py` picks manylinux for anything not
  win32) that could only fail. Tray callbacks take no arguments, or swallow the
  bool explicitly like `TrayIcon._gpu_clicked`.
- The Enable/Disable label is **pulled** from a predicate on `QMenu.aboutToShow`,
  not pushed by the app: `_on_gpu_setup_finished` returns early for a
  tray-triggered setup — the one case a push would be for — and the answer also
  changes where no signal reaches (`pywhispr disable-gpu` in a terminal, a
  directory deleted by hand). A predicate that raises counts as "off": a menu that
  will not open is the whole UI, and the worst that costs is a declined offer.
- **Two different "disables", on purpose.** The tray sets `use_gpu = false` and
  leaves the download alone, so switching back on costs nothing;
  `pywhispr disable-gpu` deletes the libraries and reclaims the disk. `use_gpu`
  outranks `use_directml` and is honoured in `directml.activate_if_enabled` and
  `onnx_backend.providers_for` — the libraries are still advertised, so refusing
  them there is what actually turns it off.
- Neither direction can take effect in-process: onnxruntime resolves a session's
  providers when it is built, which is why `cuda.verify()` needs a subprocess and
  why both paths end in a restart notice. `_run_gpu_setup` switches `use_gpu` back
  on before that check, or it would read the config, report the CPU and call a
  good install a failure.
- `gpu.turn_off` also clears `offer_gpu_setup` (or the next start offers to install
  what was just switched off) and resets `model_quantization` when it is `""` —
  the empty string is only ever written by the first-run CUDA path, and full
  precision on the CPU is the slowest combination there is.

## Debugging native crashes here

- **`lldb` attach is blocked by MDM** and **ReportCrash is disabled** (no
  `.ips` files). Don't rely on either.
- Use Python's `faulthandler` instead (in-process, no permission needed).
  `launch.py` enables it. Run the bundle binary **directly** with stderr
  redirected to a file (not via `open`, which sends stderr to the system log).
- Read the exit code: **139 = SIGSEGV** (faulthandler dumps it by default),
  **133 = SIGTRAP** (native abort in Qt/Metal/ObjC — faulthandler does NOT
  catch it by default; register it explicitly if needed).

## Hotkeys (the fiddly part)

- **Chords** (e.g. `<cmd>+<shift>+<space>`) use Carbon `RegisterEventHotKey`
  on macOS — permission-free.
- **Double-tap** (`double-tap:<alt>`) uses a Cocoa **NSEvent global monitor**
  on macOS, NOT pynput: pynput's `keyboard.Listener` builds its keycode map
  with Carbon calls off the main thread and **segfaults** alongside our Carbon
  hotkey. Double-tap needs the **Input Monitoring** permission; chords don't.
- pynput is used for both on Windows/Linux (fine there).
- Accessibility is optional: without it, text goes to the clipboard instead of
  auto-pasting.
- The app owns recording state; listeners only report events (incl. the
  double-tap hold duration for push-to-talk), so state can't desync.

## Making a release

```sh
./scripts/make-release   # bumps patch version in pyproject.toml + __init__.py,
                         # commits, tags vX.Y.Z, pushes → triggers the workflow
```

`.github/workflows/release.yml` builds the macOS `.app` (zip) and Windows MSI.
**Lesson:** parallel build jobs must NOT each create/update the release — that
races and drops assets. Build jobs upload artifacts; a single final job
publishes them together. Bump the version files for real (the MSI's in-place
upgrade depends on it).

## Packaging (`packaging/`)

- `pywhispr.spec` — PyInstaller, menu-bar-only (`LSUIElement`), ad-hoc signed.
  **`mlx.metallib` must be copied next to the relocated `libmlx.dylib`** or MLX
  crashes on load ("Failed to load the default metallib").
- `setup_msi.py` — cx_Freeze, per-user (no admin). Autostart is an HKCU `Run`
  value in a **component of our own**: the executable's component id comes from
  cx_Freeze internals (`make_id(f"_cx_executable{idx}_{executable}")`), so
  borrowing it would break the moment cx_Freeze changed. Ours carries a fixed
  GUID, the registry value as its key path, and a `AUTOSTART<>"0"` condition so
  the MSI can be run without it. Launch-on-finish is cx_Freeze's own
  `launch_on_finish` option. None of this can be built or tested off Windows —
  `msilib` is Windows-only, so `bdist_msi.finalize_options` refuses to run; what
  you *can* check locally is the table rows, by loading `setup_msi.py` by path
  with a stand-in `cx_Freeze` that records the kwargs (`tests/test_setup_msi.py`).
- **cx_Freeze is pinned in the release workflow, and that is now a correctness
  dependency, not tidiness.** The stop-the-running-app actions below sit at
  particular sequence numbers relative to cx_Freeze's own rows (401/402, and a
  `RemoveExistingProducts` it forces to 1450) and depend on the
  `REMOVEOLDVERSION`/`REMOVENEWVERSION` property names its `Upgrade` rows set.
  Those are internals: a floating version could break an upgrade path with a
  green build. Re-read `bdist_msi.py` when bumping it.
- `make_icns.sh` / `PyWhispr.ico` — icons; `scripts/make_icon.py` regenerates
  the base PNG (background removal + speech bubble).

## One instance, and upgrades (`instance.py`)

Two upgrades needed this. The Windows MSI replaces files a running `PyWhispr.exe`
still holds open, so without help it reaches `InstallValidate` and falls back to
the *files in use* dialog — which asks the user to close an app that has no
window — or, in a `/qn` install, schedules a reboot. On macOS there is no
installer: the bundle is dragged over the old one, the replacement succeeds, and
the **old process keeps running from the deleted inode**, leaving two tray icons,
two hotkey registrations (the second Carbon `RegisterEventHotKey` fails) and two
model loads.

- **Ownership is a platform primitive, never a pid file.** A pid file goes stale
  the instant a process is killed, and the guesswork that follows is how these
  guards end up either refusing to start after a crash or terminating a stranger
  that inherited the pid. Both primitives here are released by the OS however the
  process dies: on Windows a **named event** (the name exists only while a handle
  is open, and `CreateEventW` reports `ERROR_ALREADY_EXISTS` *atomically*, so two
  simultaneous launches cannot both win), on POSIX an **exclusive `flock`** on
  the state file — which is also what makes the pid *in* that file trustworthy,
  since only the lock holder ever writes it.
- `Local\`, not `Global\`: the install is per-user, another logged-on user's
  instance is none of our business, and `Global\` needs a privilege and an ACL to
  be openable at all. The price is that a per-user MSI pushed in *system* context
  (Intune) runs its custom action in session 0, where the name refers to a
  different object — the action does nothing and the files-in-use dialog is what
  the user gets, i.e. today's behaviour.
- **The quit signal is a watcher thread, not SIGTERM.** A Python signal handler
  only runs when the interpreter regains control, and an idle tray app blocked in
  `QApplication.exec()` may not give it any for minutes. The alternative is a
  `set_wakeup_fd` + `QSocketNotifier` hop or a permanent polling `QTimer`, both of
  which put Qt in here — and Qt-free is what lets `cli.py` use this module without
  importing PySide6 in a process the installer is synchronously waiting on. A
  thread that blocks until asked is the same shape on both platforms:
  `WaitForSingleObject` on Windows, `accept()` on a unix socket on POSIX.
- **`QApplication.exec()` clears the thread's `quitNow` flag on entry, so a quit
  served before the loop starts is silently lost.** Verified, not assumed: quit
  before `exec()` and the app runs on regardless. `start()` can sit in a modal
  dialog of its own (the storage offer, the GPU offer, Lite's server prompt), so
  this is reachable — hence `run_app` checking `whispr._quitting` before
  `app.exec()`. Without it the app comes back from the dead and the installer
  escalates to a kill.
- **`QApplication.quit()` *does* unwind a modal dialog's nested loop** (
  `QCoreApplication::exit` exits every event loop on the thread), and that is the
  trap rather than the fix: the code *after* `exec()` then runs, so
  `_show_settings`' `finally: self._resume_listeners()` would register a global
  hotkey for a process that is exiting. Hence `_quitting`, checked in
  `_resume_listeners` and before `_apply_settings`. `closeAllWindows()` is for
  making the UI go at once; it is not what unwinds the loop.
- **`_on_quit_requested` ends in `os._exit(0)`, and the tray's Quit does not.** By
  then `_quit` has done everything that matters — the mixer levels, the audio
  stream, the API socket, the listeners, the download in flight — but the
  transcription worker is a `ThreadPoolExecutor` whose threads are **joined at
  interpreter exit**, so a request arriving during a 20–60s model load would hold
  the process open for the whole load with an installer waiting on it.
- **Displacement is keyed on the version differing, and deliberately not on the
  executable path.** The version is what does the work: a macOS drag-upgrade
  replaces the bundle *at the same path*, so `sys.executable` is identical and
  only the version moves. Adding the path would be worse than useless — it would
  have `uv run pywhispr` silently kill the installed app and the installed app
  kill the dev run. Same version ⇒ refuse and leave the running one alone, because
  restarting it costs the user a model load for nothing.
- A refused launch is **silent**: there is no tray icon yet to speak from, and the
  app the user asked for is already running.
- `request_quit` escalates to `TerminateProcess`/`SIGKILL` after a bounded wait,
  and **escalates at once when nothing was listening** — an old build with no
  `quit` command, or a socket that could not be bound, is not something to wait
  15 seconds for. On Windows the pid is checked against the recorded image with
  `QueryFullProcessImageNameW` first, so pid reuse cannot make us kill a stranger.
- Everything is **flavour-scoped** through `config_path().parent` and
  `flavor.PRODUCT_NAME`: PyWhisprLite is the same codebase carrying the same
  version string, so a flavour-blind name would have Lite displacing a full
  install. Note `logging_setup.log_dir()` is *not* flavour-aware, which is why
  the state does not live there.
- A unix socket path is limited to ~104 bytes and `~/Library/Application
  Support/PyWhispr` already spends 68, so failing to bind is a logged warning and
  not a failure to start. The cost is a force-kill instead of a graceful one.
- **A custom action's `Source` is a Directory *table key*, not a property.**
  cx_Freeze authors that table from the build tree and nothing else, so `SystemFolder`
  — which is a real property, formats correctly in a command line, and is the obvious
  thing to name — has no row, and Windows Installer aborts the install with **error
  2727**. v0.2.23 shipped that way. Both actions use `TARGETDIR`, the only key there
  is, and the system path stays in the Target string where the installer resolves it.
- **The MSI runs `PyWhispr.exe quit` before it touches a file, and then
  `taskkill`.** Two actions, and the order is the point — see `setup_msi.py`. The
  graceful one runs the *old* build's exe, and every build before this one exits 2
  on an unknown subcommand, so for exactly one upgrade `taskkill` is the only
  thing that works; afterwards it is the cure for a wedged instance and for an old
  install that lives somewhere other than `[TARGETDIR]`. It is second so the
  graceful path gets first refusal: a killed process leaves the per-app mixer
  levels turned down, and Windows remembers those forever.
- `pywhispr quit` is dispatched in `cli.main` **before** the certificates and
  `startup.prepare()` — re-pointing the model cache and swapping in DirectML for a
  command that is about to exit is work with nowhere to land. It is also now a
  permanent part of the installer's contract: the name, and "block until the app is
  gone".
- **Never log the state directory's contents beyond pids, versions and counts** —
  same rule as everywhere else that touches the user's machine.
- Testing gotcha: `tests/test_instance.py` uses **no `unittest.mock` at all** —
  real processes, real threads, real sockets — because this is the module whose job
  is crossing those boundaries, and mocks on a second thread are what hung the
  suite before. Its `state_dir` fixture puts the socket somewhere *short*: pytest's
  `tmp_path` overflows `sun_path` and you end up testing the degradation path
  instead of the feature.
- Testing gotcha: the guard is **injected** into `PyWhisprApp` rather than reached
  as module state, which is the whole reason `isolated_app` needs no new patch —
  the suite builds apps directly, never through `run_app`, so it claims no named
  event and no socket.
