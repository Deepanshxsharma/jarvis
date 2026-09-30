"""Memory search-parameter extraction: structured output, validation and
safe failure.

The extractor decides which diary window and which graph questions a
memory-backed reply sees, so a malformed or invented answer silently
changes what the assistant can recall.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from jarvis.reply.enrichment import (
    SEARCH_PARAMS_SCHEMA,
    extract_search_params_for_memory,
    parse_search_params,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def _cfg(**over):
    base = dict(llm_chat_model="m", ollama_base_url="http://x", ollama_chat_model="m")
    base.update(over)
    return SimpleNamespace(**base)


def _run(query, responses, cfg=None):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return responses[min(len(calls), len(responses)) - 1]

    with patch("jarvis.reply.enrichment.call_llm_direct", side_effect=fake_call):
        result = extract_search_params_for_memory(query, cfg or _cfg(), "m", timeout_sec=1.0)
    return result, calls


class TestRequestShape:
    def test_requests_schema_constrained_json(self):
        _, calls = _run("what did we discuss about the garden?", ['{"keywords": ["garden"]}'])
        schema = calls[0]["json_schema"]
        assert schema is SEARCH_PARAMS_SCHEMA
        assert schema["required"] == ["keywords"]
        assert set(schema["properties"]) >= {"keywords", "questions", "time_phrase", "from", "to"}

    def test_output_budget_fits_a_full_answer(self):
        # A full answer (5 keywords, 2 questions, a time phrase and two
        # timestamps) must fit; the old 50-token cap truncated it mid-object.
        full = json.dumps({
            "keywords": ["interests", "hobbies", "preferences", "likes", "passionate"],
            "questions": ["what topics interest the user?", "what are the user's hobbies?"],
            "time_phrase": "last week",
            "from": "2026-09-21T00:00:00Z",
            "to": "2026-09-27T23:59:59Z",
        })
        _, calls = _run("q", ['{"keywords": []}'])
        # ~4 characters per token is a generous upper bound on token density.
        assert calls[0]["max_tokens"] >= len(full) // 4 * 2

    def test_uses_configured_decision_temperature(self):
        _, calls = _run("q", ['{"keywords": []}'])
        assert calls[0]["temperature"] == 0.0
        _, calls = _run("q", ['{"keywords": []}'], cfg=_cfg(llm_decision_temperature=0.4))
        assert calls[0]["temperature"] == 0.4


class TestSafeFailure:
    def test_truncated_json_retries_once_then_returns_empty(self):
        truncated = '{"keywords": ["garden", "plants"], "questions": ["what'
        result, calls = _run("q", [truncated, truncated])
        assert result == {}
        assert len(calls) == 2

    def test_truncated_first_attempt_recovers_on_retry(self):
        result, calls = _run("q", ['{"keywords": ["gar', '{"keywords": ["garden"]}'])
        assert result["keywords"] == ["garden"]
        assert len(calls) == 2

    @pytest.mark.parametrize("bad", [
        '{"keywords": "garden"}',
        '["garden"]',
        '{"questions": ["x"]}',
        "not json at all",
    ])
    def test_schema_violations_are_rejected(self, bad):
        assert parse_search_params(bad, "q", now=NOW) is None

    def test_non_string_list_items_are_dropped(self):
        params = parse_search_params(
            '{"keywords": ["garden", 3, null, " ", "Garden"], "questions": [1, "who?"]}',
            "q", now=NOW,
        )
        assert params == {"keywords": ["garden"], "questions": ["who?"]}


class TestTimeRangeGrounding:
    def _parse(self, payload, query):
        return parse_search_params(json.dumps(payload), query, now=NOW)

    def test_range_without_time_phrase_is_dropped(self):
        params = self._parse(
            {"keywords": ["garden"], "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"},
            "what did we discuss about the garden?",
        )
        assert params == {"keywords": ["garden"], "questions": []}

    def test_range_whose_phrase_is_absent_from_query_is_dropped(self):
        params = self._parse(
            {"keywords": ["garden"], "time_phrase": "yesterday",
             "from": "2026-09-29T00:00:00Z", "to": "2026-09-29T23:59:59Z"},
            "what did we discuss about the garden?",
        )
        assert "from" not in params and "to" not in params

    def test_grounded_range_is_kept_and_normalised(self):
        params = self._parse(
            {"keywords": ["garden"], "time_phrase": "Last  Week",
             "from": "2026-09-21T00:00:00+00:00", "to": "2026-09-27T23:59:59"},
            "what did we say about the garden last week?",
        )
        assert params["from"] == "2026-09-21T00:00:00Z"
        assert params["to"] == "2026-09-27T23:59:59Z"

    def test_open_ended_grounded_range_is_kept(self):
        params = self._parse(
            {"keywords": ["garden"], "time_phrase": "since monday", "from": "2026-09-28T00:00:00Z"},
            "what have we said about the garden since Monday?",
        )
        assert params["from"] == "2026-09-28T00:00:00Z"
        assert "to" not in params

    def test_inverted_range_is_dropped(self):
        params = self._parse(
            {"keywords": ["garden"], "time_phrase": "last week",
             "from": "2026-09-27T00:00:00Z", "to": "2026-09-21T00:00:00Z"},
            "garden last week",
        )
        assert "from" not in params and "to" not in params

    def test_unparseable_timestamp_drops_whole_range(self):
        params = self._parse(
            {"keywords": ["garden"], "time_phrase": "last week",
             "from": "2026-09-21T00:00:00Z", "to": "the end of last week"},
            "garden last week",
        )
        assert "from" not in params and "to" not in params

    def test_range_starting_in_the_future_is_dropped(self):
        start = NOW + timedelta(days=3)
        params = self._parse(
            {"keywords": ["garden"], "time_phrase": "next week",
             "from": start.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "garden next week",
        )
        assert "from" not in params

    def test_end_to_end_invented_range_does_not_reach_caller(self):
        invented = json.dumps({
            "keywords": ["novaforge", "technologies"],
            "from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z",
        })
        result, _ = _run("What technologies does NovaForge use?", [invented])
        assert result["keywords"] == ["novaforge", "technologies"]
        assert "from" not in result and "to" not in result


class TestBackendStructuredOutput:
    def _post_capture(self):
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.json.return_value = {"message": {"content": '{"keywords": []}'}}
        resp.raise_for_status.return_value = None
        return resp

    def test_ollama_sends_schema_as_format(self):
        from jarvis.llm.ollama import OllamaBackend

        resp = self._post_capture()
        with patch("jarvis.llm.ollama.requests.post", return_value=resp) as post:
            OllamaBackend("http://x").direct("m", "s", "u", json_schema=SEARCH_PARAMS_SCHEMA)
            assert post.call_args.kwargs["json"]["format"] == SEARCH_PARAMS_SCHEMA
            OllamaBackend("http://x").direct("m", "s", "u")
            assert "format" not in post.call_args.kwargs["json"]

    def test_openai_compatible_sends_schema_as_response_format(self):
        from jarvis.llm.openai_compatible import OpenAICompatibleBackend

        resp = self._post_capture()
        resp.json.return_value = {"choices": [{"message": {"content": '{"keywords": []}'}}]}
        with patch("jarvis.llm.openai_compatible.requests.post", return_value=resp) as post:
            OpenAICompatibleBackend("http://x/v1").direct(
                "m", "s", "u", json_schema=SEARCH_PARAMS_SCHEMA,
            )
            rf = post.call_args.kwargs["json"]["response_format"]
            assert rf["type"] == "json_schema"
            assert rf["json_schema"]["schema"] == SEARCH_PARAMS_SCHEMA


class TestDecisionTemperatureConfig:
    def test_defaults_to_zero(self, monkeypatch, tmp_path):
        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        assert load_settings().llm_decision_temperature == 0.0

    @pytest.mark.parametrize("raw, expected", [(0.3, 0.3), (-1, 0.0), ("warm", 0.0)])
    def test_user_value_is_validated(self, monkeypatch, tmp_path, raw, expected):
        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"llm_decision_temperature": raw}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        assert load_settings().llm_decision_temperature == expected

    def test_helper_tolerates_hand_built_cfg(self):
        from jarvis.llm import decision_temperature

        assert decision_temperature(SimpleNamespace()) == 0.0
        assert decision_temperature(SimpleNamespace(llm_decision_temperature=0.2)) == 0.2
        assert decision_temperature(SimpleNamespace(llm_decision_temperature=None)) == 0.0
