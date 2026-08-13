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
  `_teams_emoji.extract()` works and is tested; it simply has no caller. Every route
  was rejected for a reason worth keeping: `act` runs after the injector has already
  overwritten the clipboard; `rewrite` must be reentrant and I/O-free and also runs
  on API request threads; a `pywhispr` subcommand puts Teams into the main program's
  command surface, which is what the framework exists to prevent; and a tray entry
  needs plugins to declare menu actions generically, which nobody has asked for.
  So the store is read-only in practice — hand-editing `custom_emoji.json` works.
- The store is JSON rather than a `vocabulary.txt`-style line format because the
  fragments are hundreds of characters of markup; a hand-editable one-per-line file
  would be a fiction, though deleting an entry still works.
- **Nothing in the main program knows what an emoji is.** `_teams_emoji.py` lives
  inside `plugins/builtin/` with a leading underscore, imported only by `emoji.py`.
  An earlier version put it in `pywhispr/` and gave `cli.py` a `learn-emoji`
  subcommand — 19 emoji references in the main CLI — which contradicted this file's
  own claim that the framework knows nothing about emoji. `richclip.py` is the
  exception that proves the rule: it stays in the main package because its code is
  generic (text + spans → HTML) and `injector.py` is its real consumer.
- The stored fragments are **tenant-scoped** (image URLs fetched with the viewer's
  credentials), so the store is not shareable configuration.
- macOS `NSPasteboard` path is written from the documented API and **unverified on a
  real machine**; every failure returns False and falls back to plain text.
- Testing gotcha: `isolated_app` patches `plugins.builtin._teams_emoji.load` too, or
  the suite reads whatever the developer happens to have in their own store.

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
  cx_Freeze internals (`make_id(f"_cx_executable{idx}_{executable}")`), and CI
  installs whatever cx_Freeze is current, so borrowing it would break silently.
  Ours carries a fixed GUID, the registry value as its key path, and a
  `AUTOSTART<>"0"` condition so the MSI can be run without it. Launch-on-finish
  is cx_Freeze's own `launch_on_finish` option. None of this can be built or
  tested off Windows — `msilib` is Windows-only, so `bdist_msi.finalize_options`
  refuses to run; the most you can check locally is that the option names and
  emitted table rows are right (stub `HAS_MSILIB`/`add_data` and call
  `add_config`).
- `make_icns.sh` / `PyWhispr.ico` — icons; `scripts/make_icon.py` regenerates
  the base PNG (background removal + speech bubble).
