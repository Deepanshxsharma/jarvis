"""Stop and interrupt phrases configured by the user reach the listener."""

import json

import pytest

from jarvis.config import load_settings

pytestmark = pytest.mark.unit


def test_default_stop_and_interrupt_phrases(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

    s = load_settings()

    assert "stop" in s.stop_commands
    assert "wait" in s.interrupt_commands
    assert s.stop_command_fuzzy_ratio == 0.8


def test_user_phrases_override_defaults(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "stop_commands": [" Ruko ", "bas"],
        "interrupt_commands": ["Ek Minute"],
        "stop_command_fuzzy_ratio": 0.9,
    }), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

    s = load_settings()

    assert s.stop_commands == ["ruko", "bas"]
    assert s.interrupt_commands == ["ek minute"]
    assert s.stop_command_fuzzy_ratio == 0.9
