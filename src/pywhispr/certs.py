"""Verify TLS against the operating system's trust store, not just certifi.

Networks that TLS-inspect (Cloudflare Gateway, Zscaler and friends) hand us a
locally-issued certificate for huggingface.co. The machine itself trusts the
intercepting CA — it must, or no browser on it would work — but Python does
not: `requests` and `httpx` verify against certifi's bundled PEM list, which by
definition cannot contain a private CA. So the first-run model download dies
with CERTIFICATE_VERIFY_FAILED on a machine where every other app is fine.

`truststore` redirects verification to the platform verifier (Schannel on
Windows, SecTrust on macOS, OpenSSL's configured paths on Linux), so whatever
the machine trusts, we trust. Nothing here is specific to one CA or one vendor
and no certificate is shipped or pinned, so a rotated or replaced corporate
root needs no change to PyWhispr.

Injection is unconditional, including when the CA-bundle environment variables
are set. Standing aside for those was worse than useless on the machine this
came from: ``SSL_CERT_FILE`` pointed at a single corporate ``.cer``, which
replaces the public roots rather than adding to them, so every download failed
with "unable to get local issuer certificate" — and it only started failing when
``huggingface_hub`` 1.x moved from ``requests`` to ``httpx``, which reads
``SSL_CERT_FILE`` where ``requests`` reads ``REQUESTS_CA_BUNDLE``. Nothing is
lost by injecting anyway: truststore tries the OS store first and falls back to a
chain engine trusting whatever ``load_verify_locations`` was given, so an
explicit bundle still counts.

One thing this deliberately does not do:

* **Fail loudly.** Injection is best-effort. If truststore is missing from a
  build or the platform call fails, we log it and carry on with certifi — which
  is exactly the old behaviour, and still works everywhere that isn't
  intercepted. A tray app that refuses to start is indistinguishable from a
  crash.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# Reported, not obeyed: which of these is set decides which stack breaks when one
# of them names a bundle without the public roots, so a bug report needs to say.
CA_BUNDLE_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")

_status = "not attempted"


def trust_store_status() -> str:
    """One-line description of what verification is in force, for bug reports."""
    return _status


def use_system_certificates() -> str:
    """Route TLS verification through the OS trust store. Returns the outcome.

    Safe to call more than once and safe to call on any platform; every failure
    path degrades to certifi rather than raising.
    """
    global _status

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as exc:
        # Includes ImportError: a build that missed the hidden import should
        # still run, just without help behind an intercepting proxy.
        _status = f"certifi only (system trust store unavailable: {exc!r})"
        log.warning(
            "Could not use the system certificate store; TLS verification falls back "
            "to certifi, which may fail behind corporate TLS inspection: %r",
            exc,
        )
        return _status

    bundles = [name for name in CA_BUNDLE_ENV_VARS if os.environ.get(name)]
    _status = "system trust store (truststore)"
    if bundles:
        _status += f", plus the bundle in {', '.join(bundles)}"
    log.debug("TLS verification now uses the system certificate store")
    return _status
