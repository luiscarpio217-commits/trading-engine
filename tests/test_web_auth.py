"""Dashboard auth: basic-auth middleware and the fail-closed bind policy."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from trading_engine.dashboard.web import (PASSWORD_ENV, USERNAME_ENV, create_app,
                                          resolve_credentials, serve)

STATUS = {"broker": "paper", "positions": [], "signals": [], "trades": [],
          "stats": {}, "account": {}, "day": {}}


def provider():
    return STATUS


@pytest.fixture
def no_env_creds(monkeypatch):
    monkeypatch.delenv(USERNAME_ENV, raising=False)
    monkeypatch.delenv(PASSWORD_ENV, raising=False)


class TestBasicAuthMiddleware:
    def test_everything_requires_auth(self):
        client = TestClient(create_app(provider, "trader", "s3cret"))
        for path in ("/", "/api/status", "/api/positions", "/api/trades",
                     "/api/signals", "/api/stats", "/docs", "/openapi.json"):
            r = client.get(path)
            assert r.status_code == 401, path
            assert r.headers.get("www-authenticate", "").startswith("Basic")

    def test_correct_credentials_pass(self):
        client = TestClient(create_app(provider, "trader", "s3cret"))
        assert client.get("/", auth=("trader", "s3cret")).status_code == 200
        r = client.get("/api/status", auth=("trader", "s3cret"))
        assert r.status_code == 200 and r.json()["broker"] == "paper"

    def test_wrong_credentials_rejected(self):
        client = TestClient(create_app(provider, "trader", "s3cret"))
        assert client.get("/", auth=("trader", "wrong")).status_code == 401
        assert client.get("/", auth=("intruder", "s3cret")).status_code == 401
        # garbage header must not crash the middleware
        assert client.get("/", headers={"Authorization": "Basic %%%%"}).status_code == 401
        assert client.get("/", headers={"Authorization": "Bearer x"}).status_code == 401

    def test_no_credentials_configured_is_open(self):
        """Local development mode: create_app without creds stays open."""
        client = TestClient(create_app(provider))
        assert client.get("/api/status").status_code == 200


class TestServePolicy:
    def test_refuses_public_bind_without_credentials(self, no_env_creds):
        with pytest.raises(RuntimeError, match="refusing to bind"):
            serve(provider, host="0.0.0.0", port=8000)
        with pytest.raises(RuntimeError, match="DASHBOARD_USERNAME"):
            serve(provider, host="10.0.0.5", port=8000)

    def test_explicit_partial_credentials_still_fail_closed(self, no_env_creds):
        with pytest.raises(RuntimeError, match="refusing to bind"):
            serve(provider, host="0.0.0.0", port=8000, username="user", password=None)

    def test_resolve_credentials_from_env(self, monkeypatch):
        monkeypatch.setenv(USERNAME_ENV, "trader")
        monkeypatch.setenv(PASSWORD_ENV, "pw")
        assert resolve_credentials() == ("trader", "pw")
        monkeypatch.setenv(PASSWORD_ENV, "   ")  # blank counts as unset
        assert resolve_credentials() == ("trader", None)

    def test_env_credentials_enforced_end_to_end(self, monkeypatch):
        """serve() wires env creds into the app (checked via create_app parity)."""
        monkeypatch.setenv(USERNAME_ENV, "trader")
        monkeypatch.setenv(PASSWORD_ENV, "pw")
        user, password = resolve_credentials()
        client = TestClient(create_app(provider, user, password))
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status", auth=("trader", "pw")).status_code == 200
