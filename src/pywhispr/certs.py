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

Two things this deliberately does not do:

* **Override an explicit choice.** If any of the CA-bundle environment
  variables is set, the user has configured certificates on purpose (it is what
  the README tells them to do) and we leave verification alone rather than
  quietly changing it underneath them.
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

# SSL_CERT_FILE is read by Python's ssl module, so it covers httpx (and
# therefore huggingface_hub's downloads); REQUESTS_CA_BUNDLE and CURL_CA_BUNDLE
# are honoured by requests alone. Any of them means "certificates are handled".
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

    overrides = [name for name in CA_BUNDLE_ENV_VARS if os.environ.get(name)]
    if overrides:
        _status = f"certifi/explicit bundle ({', '.join(overrides)} set)"
        log.debug("Leaving TLS verification alone: %s", _status)
        return _status

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

    _status = "system trust store (truststore)"
    log.debug("TLS verification now uses the system certificate store")
    return _status
