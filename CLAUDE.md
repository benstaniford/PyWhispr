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
  Setting either makes `certs.py` stand aside. Once the model is cached, normal
  runs work offline.
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

## Filler removal (`filler.py`)

Runs **first** in the transcript pipeline (fillers → vocabulary → join): each
pass changes what the next one sees, and the join decides about the opening
word, so it has to see the word that will actually be first. It is also applied
*before* `_on_transcribed`'s empty check, so a recording of nothing but "um"
falls into the existing "nothing to insert" path instead of pasting a full stop.

- **Deletion only**, and `app._cleaned` proves it with
  `is_deletion_only` — every alphanumeric of the output appears in order in the
  input, case-insensitively, so re-capitalisation is allowed and a substituted
  or invented word is not. Same reason as `_joined`/`_corrected`: the audio is
  gone by now.
- The list is a **closed list of noises**, not "words that look like filler".
  Absent on purpose: `like`, `right`, `so`, `well`, `mean`, `ah`, `oh`, `eh`,
  `hmm`, `mm` (ordinary words or things people type deliberately), `err` (to
  err). Three guards beyond the list, each from a case that produced real word
  loss: `_token_key` refuses an **all-caps token** (`ER`, `UM` and `ERM` are a
  department, a system and a report — casefolding is what makes them look like
  fillers); `PROTECTED_FOLLOWERS` refuses `uh`/`um` before `huh` or `oh` (*uh
  huh* means yes, *uh oh* means trouble — only the hyphenated spellings are safe
  for free, since `_WORD` keeps those one token); and `extra_filler_words` /
  `keep_filler_words` are the out for everything else.
- Deleting a word is the easy half; the **seam** is the work, and every branch
  in the loop is a case where the naive deletion read worse than the "um" did:
  mid-sentence ("I think, uh, we go" — tidy on the right), sentence-final
  ("fine, um." — nothing on the right to tidy, so take the left), a whole
  sentence of filler ("Hello. Um. Right." — the full stop was the filler's, so it
  goes too or it doubles up), a bracket or quote **pair** that was only holding
  the filler ("(um)" → "()" otherwise), a **dash** pair (one dash stays — the
  dashes are the sentence's punctuation, and deleting both merged two
  parentheticals), and `tidy()` for the blank line, doubled space or stray mark a
  deletion leaves. `_opens_sentence` is wider than `join.is_mid_sentence`: an
  opener, a quote or a list marker also wants the capital handed on.
- `_paired_mark` exists because **a straight quote or dash is both halves of its
  own pair**. In `He said "go" um "now".` the mark in front of the filler is
  *closing* the previous quotation, and pairing it with the next one fused "go"
  and "now" into a single word. Quotes therefore only count as opening where an
  opening quote can stand; dashes don't nest, so they always pair. `tidy()`'s
  last line is the backstop: two words never fuse, whatever the branches do.
- The **comma pair** is the one genuinely ambiguous call. Two commas around a
  filler are usually its pauses ("I think, um, we go" → "I think we go"), but in
  a list they are the sentence's ("milk, um, eggs" must keep its comma), and
  nothing local decides it. `was_a_pause` uses `PAUSE_WORDS`, a short *positive*
  list of words a hesitation follows, so the failure is a surplus comma that
  reads as the pause it was, never a lost separator. `CONTINUATION_WORDS` was
  tried here first and is the wrong list — it answers "may this word be
  lower-cased" and includes `this`, `one` and `yes`, which end list items.
  Three exceptions on top of the list, each from a case that read wrongly:
  `SUBORDINATORS` keeps the comma when the clause opened with `if`/`while`/…,
  because "If you do, um, tell me" → "If you do tell me" changes the meaning
  (`_clause_opener` skips a leading `and`/`but`/`even`, which otherwise hid the
  subordinator, and scans *backwards* — splitting the whole prefix made the pass
  quadratic); an object pronoun before and a subject pronoun after means the
  comma was splicing two clauses ("I saw it, um, I left"); and a lone capital
  letter is a label, not the article `a` ("plan A, um, plan B") — except
  sentence-initially, where it can only be the article.
- Known limits, all deliberate: an ASCII `-` is not in `_PAIRS` (bullets, minus
  signs and ranges outweigh the rare `- um -`), a filler alone on a bulleted line
  leaves the bullet, a comma inside the clause hides the subordinator from
  `_clause_opener` ("If, as you say, you can, um, come"), two imperatives joined
  by a comma after a preposition fuse ("Come on in, um, sit down"), and a repeated
  word around a deleted filler stays repeated — this pass deletes, it does not
  rewrite.

## Conversational filler was tried and dropped

A second tier removing *you know*, *I mean*, *basically*, *sort of* was built,
reviewed and deleted (it is in the session history if ever wanted). The lesson:
the difference between "you know that I left" and "it's you know complicated" is
meaning, and no closed list, punctuation rule or per-phrase policy draws it
reliably — every gate strict enough to be safe missed most real hesitations,
because the model punctuates them only sometimes. Wispr Flow does this with a
fine-tuned LLM cleanup pass in the cloud; that is the shape of the answer, and it
is a different project from a word list. `extra_filler_words` covers anyone who
wants a specific phrase gone unconditionally.

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
