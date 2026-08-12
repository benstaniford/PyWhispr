"""Plugins: run something, or change something, when a dictation says a phrase.

The transcript passes (:mod:`pywhispr.scratch`, :mod:`pywhispr.filler`,
:mod:`pywhispr.vocab`, :mod:`pywhispr.join`) each do one fixed job. This is the
open-ended one: a phrase the user says, and whatever they want to happen when
they say it. The worked example is emoji — "thumbs up emoji" becoming one
character — but nothing in the mechanism knows about emoji.

Where it sits::

    scratch → filler → vocab → PLUGINS → join

After vocab, so a trigger benefits from the user's own spellings; before join,
because a rewrite can change the transcript's opening word and join has to see
the final one.

The layers, smallest first:

- :mod:`pywhispr.plugins.api` — the dataclasses a plugin imports.
- :mod:`pywhispr.plugins.engine` — trigger matching, claim validation, splicing.
- :mod:`pywhispr.plugins.registry` — finding and loading plugins.
- :mod:`pywhispr.plugins.actions` — the thread side effects run on.

Nothing is re-exported here on purpose. ``api`` and ``engine`` depend on the
standard library and nothing else, and a re-export would drag ``registry`` — and
through it ``config`` and ``platformdirs`` — into every plugin that imports a
dataclass. Import from the submodule, as the rest of the codebase does with
``stt`` and ``ui``.
"""
