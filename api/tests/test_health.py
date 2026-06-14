"""
api/tests/test_health.py
Tests for GET /health and GET /ready endpoints.
"""
from __future__ import annotations

import pytest
from api.tests.conftest import _need_checkpoint


@_need_checkpoint
class TestHealth:

    def test_health_returns_200(self, client) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_content_type(self, client) -> None:
        r = client.get("/health")
        assert "application/json" in r.headers["content-type"]


@_need_checkpoint
class TestReady:

    def test_ready_returns_200(self, client) -> None:
        r = client.get("/ready")
        assert r.status_code == 200

    def test_ready_is_true_when_model_loaded(self, client) -> None:
        body = client.get("/ready").json()
        assert body["ready"] is True
        assert body["model_loaded"] is True

    def test_ready_has_checkpoint_info(self, client) -> None:
        body = client.get("/ready").json()
        assert isinstance(body["checkpoint_epoch"], int)
        assert isinstance(body["checkpoint_val_f1"], float)
        assert body["checkpoint_epoch"] == 16       # epoch saved in checkpoint

    def test_ready_has_model_version(self, client) -> None:
        body = client.get("/ready").json()
        assert "model_version" in body
        assert isinstance(body["model_version"], str)
        assert len(body["model_version"]) > 0
