import sys
import types

import pytest

from pywhispr import certs


@pytest.fixture
def no_ca_env(monkeypatch):
    """A machine with no CA overrides set (the developer's may well have them)."""
    for name in certs.CA_BUNDLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_truststore(monkeypatch):
    """A stand-in for truststore.

    The real ``inject_into_ssl()`` patches ``ssl`` process-wide, which would
    leak out of the test and change how every later test verifies TLS.
    """
    calls = []
    module = types.ModuleType("truststore")
    module.inject_into_ssl = lambda: calls.append("injected")
    monkeypatch.setitem(sys.modules, "truststore", module)
    monkeypatch.setattr(certs, "_status", "not attempted")
    return calls


def test_uses_the_system_trust_store(no_ca_env, fake_truststore):
    status = certs.use_system_certificates()

    assert fake_truststore == ["injected"]
    assert "system trust store" in status
    assert certs.trust_store_status() == status


@pytest.mark.parametrize("var", certs.CA_BUNDLE_ENV_VARS)
def test_an_explicit_ca_bundle_wins(no_ca_env, fake_truststore, monkeypatch, var, tmp_path):
    """A user who configured certs deliberately keeps the behaviour they set up."""
    monkeypatch.setenv(var, str(tmp_path / "ca-bundle.pem"))

    status = certs.use_system_certificates()

    assert fake_truststore == []
    assert var in status


def test_a_broken_platform_verifier_falls_back_to_certifi(no_ca_env, monkeypatch):
    """Losing the trust store must not stop the app: it still runs off-proxy."""
    module = types.ModuleType("truststore")

    def explode():
        raise OSError("Schannel said no")

    module.inject_into_ssl = explode
    monkeypatch.setitem(sys.modules, "truststore", module)
    monkeypatch.setattr(certs, "_status", "not attempted")

    status = certs.use_system_certificates()

    assert "certifi only" in status
    assert "Schannel said no" in status


def test_a_missing_truststore_falls_back_to_certifi(no_ca_env, monkeypatch):
    """The packaged builds import it lazily, so a bad build must degrade, not crash."""
    monkeypatch.setitem(sys.modules, "truststore", None)  # import raises ImportError
    monkeypatch.setattr(certs, "_status", "not attempted")

    status = certs.use_system_certificates()

    assert "certifi only" in status


def test_status_is_reported_before_anything_is_attempted(monkeypatch):
    monkeypatch.setattr(certs, "_status", "not attempted")
    assert certs.trust_store_status() == "not attempted"


def test_the_environment_report_names_the_verification_in_force(no_ca_env, fake_truststore):
    from pywhispr.logging_setup import environment_report

    certs.use_system_certificates()

    assert any("tls verification: system trust store" in line for line in environment_report())
