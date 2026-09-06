"""
Tests for the model version registry — the mechanism DESIGN.md §6.5's
rollback story is supposed to be backed by. If `resolve_model_dir` can't
correctly distinguish "pin an explicit version" from "follow current.json",
rollback is prose, not a lever anyone could actually pull.
"""

import pytest

from src.registry import current_version, list_versions, new_version, resolve_model_dir, save_pointer


def test_new_version_is_deterministic_when_explicit():
    assert new_version("v1") == "v1"


def test_new_version_generates_something_when_not_explicit(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    v = new_version(None)
    assert v  # non-empty, sortable timestamp-shaped string
    assert v[:4].isdigit()


def test_pointer_round_trip(tmp_path):
    save_pointer(tmp_path, "20260101T000000Z")
    assert current_version(tmp_path) == "20260101T000000Z"

    save_pointer(tmp_path, "20260102T000000Z")
    assert current_version(tmp_path) == "20260102T000000Z"


def test_resolve_missing_pointer_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        current_version(tmp_path)


def test_resolve_follows_pointer_by_default(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    save_pointer(tmp_path, "v1")
    assert resolve_model_dir(tmp_path) == tmp_path / "v1"


def test_explicit_version_overrides_pointer_the_rollback_lever(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    save_pointer(tmp_path, "v2")  # "current" is v2 ...
    assert resolve_model_dir(tmp_path, version="v1") == tmp_path / "v1"  # ... but v1 is pinned


def test_env_var_also_overrides_pointer(tmp_path, monkeypatch):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    save_pointer(tmp_path, "v2")
    monkeypatch.setenv("BOOKING_CURVE_MODEL_VERSION", "v1")
    assert resolve_model_dir(tmp_path) == tmp_path / "v1"


def test_resolve_nonexistent_version_raises(tmp_path):
    save_pointer(tmp_path, "ghost-version")
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(tmp_path)


def test_list_versions_sorted_and_empty_base_is_safe(tmp_path):
    assert list_versions(tmp_path / "does-not-exist") == []
    (tmp_path / "20260102T000000Z").mkdir()
    (tmp_path / "20260101T000000Z").mkdir()
    assert list_versions(tmp_path) == ["20260101T000000Z", "20260102T000000Z"]
