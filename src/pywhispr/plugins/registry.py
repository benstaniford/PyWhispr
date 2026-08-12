"""Finding plugins: the ones we ship, and the ones the user wrote.

Two sources, and they load differently on purpose:

- **Built-ins** live in :mod:`pywhispr.plugins.builtin` and are listed in
  :data:`BUILTINS` as ordinary module names, imported with ``import_module``.
  Not discovered by scanning the package: PyInstaller and cx_Freeze find imports
  by reading the source, so a plugin only ever named at runtime would be missing
  from both packaged builds — the kind of bug that only shows up after release.
- **User plugins** are ``*.py`` files in ``<config dir>/plugins``, loaded by path.
  One try/except each, so a file with a syntax error costs its own plugin and
  nothing else.

Loading happens once, at startup. There is no reload: a plugin that started a
thread or opened a handle cannot be un-imported, so "reload" would mean leaking
the old one and hoping. The tray opens the folder and a restart applies changes,
which is at least honest about what is happening.

**A user plugin is arbitrary code**, run at every startup with the user's own
privileges — the same trust as their shell profile. That is fine for plugins they
wrote or read, and it is why nothing here ever downloads or installs one.

Under a packaged build a plugin may import the standard library and ``pywhispr``,
and nothing else: third-party packages are not in the bundle, so an ``import
requests`` fails there while working in a checkout.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path

from platformdirs import user_config_dir

from pywhispr.config import APP_NAME, Config
from pywhispr.plugins.api import Trigger
from pywhispr.plugins.engine import Plugin, compile_patterns

log = logging.getLogger(__name__)

# Shipped plugins, in load order. Named statically so the packaged builds see them.
BUILTINS: tuple[str, ...] = ("emoji",)

MODULE_PREFIX = "pywhispr_plugin_"


def plugins_dir() -> Path:
    return Path(user_config_dir(APP_NAME)) / "plugins"


def _plugin_from_module(module, name: str, source: str) -> Plugin | None:
    """Read a module's TRIGGERS/rewrite/act into a Plugin, or None if it has none.

    Duck-typed rather than subclass-based: a plugin is a module with a couple of
    module-level names, which is the smallest thing that can be written in a text
    editor and dropped in a folder.
    """
    triggers = getattr(module, "TRIGGERS", None)
    if not triggers:
        log.warning("Plugin %r declares no TRIGGERS; ignoring it", name)
        return None
    if isinstance(triggers, Trigger):
        triggers = (triggers,)
    try:
        triggers = tuple(triggers)
    except TypeError:
        log.warning("Plugin %r has a TRIGGERS that is not a sequence; ignoring it", name)
        return None
    if not all(isinstance(trigger, Trigger) for trigger in triggers):
        log.warning("Plugin %r has TRIGGERS that are not Trigger objects; ignoring it", name)
        return None

    rewrite = getattr(module, "rewrite", None)
    act = getattr(module, "act", None)
    if rewrite is None and act is None:
        log.warning("Plugin %r implements neither rewrite nor act; ignoring it", name)
        return None
    if rewrite is not None and not callable(rewrite):
        log.warning("Plugin %r has a rewrite that is not callable; ignoring it", name)
        return None
    if act is not None and not callable(act):
        log.warning("Plugin %r has an act that is not callable; ignoring it", name)
        act = None

    patterns = compile_patterns(triggers)
    if not patterns:
        log.warning("Plugin %r has no usable trigger; ignoring it", name)
        return None

    return Plugin(
        name=getattr(module, "NAME", name) or name,
        triggers=triggers,
        rewrite=rewrite,
        act=act,
        source=source,
        patterns=patterns,
    )


def _load_builtin(name: str) -> Plugin | None:
    try:
        module = importlib.import_module(f"pywhispr.plugins.builtin.{name}")
    except Exception:
        log.exception("Built-in plugin %r could not be imported", name)
        return None
    return _plugin_from_module(module, name, source="builtin")


def _load_file(path: Path) -> Plugin | None:
    """Import one user plugin by path, under a name that cannot clash."""
    module_name = f"{MODULE_PREFIX}{path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            log.warning("Plugin file %s could not be loaded", path.name)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        # A broken plugin must cost the user that plugin, not their dictation.
        log.exception("Plugin file %s failed to import", path.name)
        return None
    return _plugin_from_module(module, path.stem, source=path.name)


def _is_enabled(cfg: Config, name: str) -> bool:
    """Per-plugin switch from ``[plugins.<name>]``; absent means on.

    Discovering a plugin at all means someone put it there, so the default is to
    run it. A malformed section counts as on for the same reason a raising GPU
    predicate counts as off: whichever direction is recoverable from.
    """
    settings = (cfg.plugins or {}).get(name)
    if isinstance(settings, dict) and "enabled" in settings:
        return bool(settings["enabled"])
    return True


def load_plugins(cfg: Config) -> list[Plugin]:
    """Every enabled plugin, built-ins first. Never raises.

    Built-ins first so their claims win a tie at the same position, and so a user
    plugin loaded later is never shadowed by one we add in a future release.
    """
    if not cfg.plugins_enabled:
        log.info("Plugins are switched off")
        return []

    plugins: list[Plugin] = []
    for name in BUILTINS:
        if not _is_enabled(cfg, name):
            log.debug("Built-in plugin %r is disabled in the config", name)
            continue
        plugin = _load_builtin(name)
        if plugin is not None:
            plugins.append(plugin)

    directory = plugins_dir()
    try:
        files = sorted(directory.glob("*.py")) if directory.is_dir() else []
    except OSError:
        log.exception("Could not list the plugins folder at %s", directory)
        files = []
    for path in files:
        if path.name.startswith("_"):
            continue  # _helpers.py and friends are imports, not plugins
        if not _is_enabled(cfg, path.stem):
            log.debug("Plugin %r is disabled in the config", path.stem)
            continue
        plugin = _load_file(path)
        if plugin is not None:
            plugins.append(plugin)

    if plugins:
        log.info(
            "Loaded %d plugin(s): %s",
            len(plugins),
            ", ".join(f"{p.name} ({p.source})" for p in plugins),
        )
    return plugins


# -- the folder itself --------------------------------------------------------

# Written next to the user's plugins the first time the folder is opened, so the
# contract is readable without going to the README. Not a .py file: only *.py is
# loaded, and an example that ran itself would be a surprise.
README = '''\
PyWhispr plugins
================

Drop a .py file in this folder and restart PyWhispr. A plugin is a module with a
TRIGGERS list and either or both of two functions:

    rewrite(match) -> Rewrite | None    change the text
    act(match) -> None                  do something

rewrite() runs between transcription and paste, on the GUI thread, so it must be
quick and must not do I/O. act() runs on its own thread once the text has been
inserted, so it may take as long as it likes.

Returning None from rewrite() means "those words were not meant for me", and the
transcript is left exactly as it was. A plugin with a rewrite only gets its act()
call when that rewrite claimed something.

A worked example — say "twenty five degrees temperature" and get "25 C":

    from pywhispr.plugins.api import Match, Rewrite, Trigger

    NAME = "temperature"
    TRIGGERS = (Trigger(phrase="temperature", at_segment_end=True),)

    def rewrite(match: Match) -> Rewrite | None:
        if not match.words_before:
            return None
        number = match.words_before[-1].text
        if not number.isdigit():
            return None                       # not for us; leave the text alone
        return match.claim_from(match.words_before[-1], f"{number} C")

Notes
-----

* You cannot change text from an earlier dictation, only the transcript being
  inserted now. PyWhispr never sends Backspace to another application.
* A claim is limited to the words in match.words_before / match.words_after.
* act() must not touch the user interface, and is never run for transcription
  requests that arrived over the network API.
* Under the packaged app you can import the standard library and pywhispr, but
  not third-party packages: they are not in the bundle.
* Anything a plugin does wrong — an exception, a bad span, taking too long — is
  logged and skipped, and the transcript survives untouched.
* Turn one off with this in config.toml:

      [plugins.temperature]
      enabled = false
'''


def ensure_plugins_dir() -> Path:
    """The plugins folder, created with its README if it was not there.

    Called when the user asks to open it, not at startup: an empty folder full of
    explanation is only useful to someone who went looking for it.
    """
    directory = plugins_dir()
    directory.mkdir(parents=True, exist_ok=True)
    readme = directory / "README.txt"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    return directory
