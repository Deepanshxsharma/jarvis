"""Memory search-parameter extraction: structured output, validation, time
resolution and safe failure.

The extractor decides which diary window and which graph questions a
memory-backed reply sees, so a malformed or invented answer silently
changes what the assistant can recall. The model only classifies a time
expression it quotes from the query; the calendar arithmetic happens in
code, in the user's local timezone.
"""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from jarvis.reply.enrichment import (
    SEARCH_PARAMS_SCHEMA,
    _SEARCH_PARAMS_MAX_TOKENS,
    extract_search_params_for_memory,
    parse_search_params,
    resolve_time_range,
)

pytestmark = pytest.mark.unit

IST = ZoneInfo("Asia/Kolkata")
# Wednesday afternoon in a UTC+05:30 zone, so a local day never lines up
# with a UTC day.
NOW = datetime(2026, 9, 30, 16, 30, tzinfo=IST)


def _cfg(**over):
    base = dict(llm_chat_model="m", ollama_base_url="http://x", ollama_chat_model="m")
    base.update(over)
    return SimpleNamespace(**base)


def _run(query, responses, cfg=None, now_local=NOW):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return responses[min(len(calls), len(responses)) - 1]

    with patch("jarvis.reply.enrichment.call_llm_direct", side_effect=fake_call):
        result = extract_search_params_for_memory(
            query, cfg or _cfg(), "m", timeout_sec=1.0, now_local=now_local,
        )
    return result, calls


def _parse(payload, query, now=NOW):
    return parse_search_params(json.dumps(payload), query, now_local=now)


def _range(time_obj, query, now=NOW):
    return resolve_time_range(time_obj, query, now)


class TestRequestShape:
    def test_requests_schema_constrained_json(self):
        _, calls = _run("what did we discuss about the garden?", ['{"keywords": ["garden"]}'])
        schema = calls[0]["json_schema"]
        assert schema is SEARCH_PARAMS_SCHEMA
        assert schema["required"] == ["keywords"]
        time_schema = schema["properties"]["time"]
        assert time_schema["required"] == ["phrase", "unit"]
        assert "monday" in time_schema["properties"]["unit"]["enum"]
        assert time_schema["properties"]["offset"]["type"] == "integer"

    def test_model_is_never_asked_for_timestamps(self):
        # Calendar arithmetic is the part small models got wrong ("tomorrow"
        # resolved to today); the schema leaves no field for it.
        assert not {"from", "to"} & set(SEARCH_PARAMS_SCHEMA["properties"])

    def test_output_budget_fits_the_largest_answer_the_schema_allows(self):
        # G: a long statement must never truncate valid output. Build the
        # biggest object the schema permits and check the token budget
        # covers it at a pessimistic 3 characters per token.
        props = SEARCH_PARAMS_SCHEMA["properties"]
        kw, q = props["keywords"], props["questions"]
        t = props["time"]["properties"]
        largest = json.dumps({
            "keywords": ["k" * kw["items"]["maxLength"]] * kw["maxItems"],
            "questions": ["q" * q["items"]["maxLength"]] * q["maxItems"],
            "time": {
                "phrase": "p" * t["phrase"]["maxLength"], "unit": "wednesday",
                "offset": -100, "date": "2026-09-30", "until_now": False,
            },
        })
        assert _SEARCH_PARAMS_MAX_TOKENS >= len(largest) / 3
        _, calls = _run("q", ['{"keywords": []}'])
        assert calls[0]["max_tokens"] == _SEARCH_PARAMS_MAX_TOKENS

    def test_uses_configured_decision_temperature(self):
        _, calls = _run("q", ['{"keywords": []}'])
        assert calls[0]["temperature"] == 0.0
        _, calls = _run("q", ['{"keywords": []}'], cfg=_cfg(llm_decision_temperature=0.4))
        assert calls[0]["temperature"] == 0.4

    def test_identical_queries_send_identical_requests(self):
        # J: with temperature 0 the only remaining source of variation would
        # be the request itself, so it must be byte-identical across calls.
        _, first = _run("did I finish the parking api?", ['{"keywords": ["api"]}'])
        _, second = _run("did I finish the parking api?", ['{"keywords": ["api"]}'])
        assert first == second


class TestPersonalStatements:
    def test_a_statement_keeps_keywords_and_has_no_range(self):
        result, _ = _run("I am working on NovaForge.", ['{"keywords": ["novaforge", "project"]}'])
        assert result == {"keywords": ["novaforge", "project"], "questions": []}

    def test_b_technology_fact(self):
        result, _ = _run(
            "NovaForge uses PHP and MySQL.",
            ['{"keywords": ["novaforge", "php", "mysql", "database"]}'],
        )
        assert result["keywords"] == ["novaforge", "php", "mysql", "database"]
        assert "from" not in result and "to" not in result

    def test_e_no_time_mentioned_never_gets_a_range(self):
        result, _ = _run(
            "I am studying computer networks.",
            ['{"keywords": ["computer networks", "studying"]}'],
        )
        assert "from" not in result and "to" not in result

    def test_e_invented_time_quote_is_dropped(self):
        # Observed on gemma4:e2b: the model quoted "now" for a query that
        # names no time. The quote is not in the query, so no range.
        payload = {"keywords": ["computer networks"],
                   "time": {"phrase": "now", "unit": "day", "offset": 0}}
        params = _parse(payload, "I am studying computer networks.")
        assert params == {"keywords": ["computer networks"], "questions": []}

    def test_empty_keywords_is_the_skip_representation(self):
        result, _ = _run("what time is it?", ['{"keywords": []}'])
        assert result == {"keywords": [], "questions": []}


class TestTimeResolution:
    def test_d_yesterday_is_the_previous_local_day(self):
        got = _range({"phrase": "Yesterday", "unit": "day", "offset": -1},
                     "Yesterday I finished the parking API.")
        assert got == {"from": "2026-09-28T18:30:00Z", "to": "2026-09-29T18:29:59Z"}

    def test_today_runs_from_local_midnight(self):
        got = _range({"phrase": "today", "unit": "day"}, "what did I say today?")
        assert got == {"from": "2026-09-29T18:30:00Z", "to": "2026-09-30T18:29:59Z"}

    def test_c_tomorrow_resolves_to_the_next_day_and_is_not_a_diary_filter(self):
        # The diary only holds past conversations, so a period that has not
        # started yet can match nothing and would hide every real entry.
        time_obj = {"phrase": "tomorrow", "unit": "day", "offset": 1}
        query = "I have a presentation tomorrow."
        assert _range(time_obj, query) == {}
        params = _parse({"keywords": ["presentation"], "time": time_obj}, query)
        assert params == {"keywords": ["presentation"], "questions": []}

    def test_future_rejection_does_not_depend_on_the_hour(self):
        time_obj = {"phrase": "tomorrow", "unit": "day", "offset": 1}
        for hour in (0, 5, 9, 16, 23):
            now = NOW.replace(hour=hour, minute=5)
            assert _range(time_obj, "a meeting tomorrow", now) == {}

    def test_last_week_is_the_previous_monday_to_sunday(self):
        got = _range({"phrase": "last week", "unit": "week", "offset": -1},
                     "what did we cover last week?")
        assert got == {"from": "2026-09-20T18:30:00Z", "to": "2026-09-27T18:29:59Z"}

    def test_this_week_spanning_now_is_kept(self):
        got = _range({"phrase": "this week", "unit": "week", "offset": 0},
                     "what have we discussed this week?")
        assert got["from"] == "2026-09-27T18:30:00Z"

    def test_last_month_and_last_year(self):
        assert _range({"phrase": "last month", "unit": "month", "offset": -1},
                      "notes from last month") == {
            "from": "2026-07-31T18:30:00Z", "to": "2026-08-31T18:29:59Z"}
        assert _range({"phrase": "last year", "unit": "year", "offset": -1},
                      "trips last year") == {
            "from": "2024-12-31T18:30:00Z", "to": "2025-12-31T18:29:59Z"}

    def test_month_offset_crosses_year_boundary(self):
        now = datetime(2026, 1, 15, 12, 0, tzinfo=IST)
        got = _range({"phrase": "two months ago", "unit": "month", "offset": -2},
                     "what did we plan two months ago?", now)
        assert got == {"from": "2025-10-31T18:30:00Z", "to": "2025-11-30T18:29:59Z"}

    @pytest.mark.parametrize("offset, day", [(-1, 28), (0, 28), (-2, 21)])
    def test_named_weekday_in_the_past(self, offset, day):
        got = _range({"phrase": "Monday", "unit": "monday", "offset": offset},
                     "what did we talk about on Monday?")
        assert got["from"] == datetime(2026, 9, day, tzinfo=IST).astimezone(
            ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_previous_weekday_when_today_is_that_weekday(self):
        wednesday = {"phrase": "Wednesday", "unit": "wednesday", "offset": -1}
        got = _range(wednesday, "what did I say on Wednesday?")
        assert got["from"] == "2026-09-22T18:30:00Z"  # 23 Sep, a week ago

    def test_upcoming_weekday_is_future(self):
        assert _range({"phrase": "next Friday", "unit": "friday", "offset": 1},
                      "presenting next Friday") == {}

    def test_until_now_runs_to_the_current_moment(self):
        got = _range({"phrase": "last 3 days", "unit": "day", "offset": -3, "until_now": True},
                     "anything from the last 3 days about cars?")
        assert got == {"from": "2026-09-26T18:30:00Z", "to": "2026-09-30T11:00:00Z"}

    def test_explicit_date_and_month(self):
        assert _range({"phrase": "12 March", "unit": "day", "date": "2026-03-12"},
                      "what did we decide on 12 March?") == {
            "from": "2026-03-11T18:30:00Z", "to": "2026-03-12T18:29:59Z"}
        assert _range({"phrase": "in March", "unit": "month", "date": "2026-03-01"},
                      "what did we plan in March?") == {
            "from": "2026-02-28T18:30:00Z", "to": "2026-03-31T18:29:59Z"}

    def test_utc_fallback_when_no_local_time_is_known(self):
        utc_now = datetime(2026, 9, 30, 12, 0, tzinfo=ZoneInfo("UTC"))
        got = _range({"phrase": "yesterday", "unit": "day", "offset": -1},
                     "what did I eat yesterday?", utc_now)
        assert got == {"from": "2026-09-29T00:00:00Z", "to": "2026-09-29T23:59:59Z"}

    def test_time_words_are_removed_from_keywords(self):
        params = _parse(
            {"keywords": ["plan", "next week"],
             "time": {"phrase": "next week", "unit": "week", "offset": 1}},
            "what's on my plan for next week?",
        )
        assert params["keywords"] == ["plan"]


class TestInvalidTimeReferences:
    @pytest.mark.parametrize("time_obj", [
        {"phrase": "last week", "unit": "fortnight", "offset": -1},     # unit outside the enum
        {"phrase": "last week", "unit": "week", "offset": "-1"},        # offset not an integer
        {"phrase": "last week", "unit": "week", "offset": True},        # bool is not an offset
        {"phrase": "on 30 February", "unit": "day", "date": "2026-02-30"},  # impossible date
        {"phrase": "on 12 March", "unit": "day", "date": "12/03/2026"},     # wrong date format
        {"phrase": "12 March", "unit": "day", "date": "2026-03-12", "offset": -1},  # date and offset
        {"phrase": "Monday", "unit": "monday", "date": "2026-09-28"},   # weekday with a date
        {"phrase": "last week", "unit": "week", "offset": -100000},     # absurd distance
        {"phrase": "next 3 days", "unit": "day", "offset": 3, "until_now": True},  # future until now
        {"phrase": "last week", "unit": "week", "offset": -1, "until_now": "yes"},
        {"phrase": "", "unit": "day", "offset": -1},                    # empty quote
        {"unit": "day", "offset": -1},                                  # no quote
        "yesterday",                                                    # not an object
    ])
    def test_i_invalid_references_produce_no_range(self, time_obj):
        assert _range(time_obj, "what did we cover last week yesterday on 12 March "
                                "on 30 February Monday next 3 days?") == {}

    def test_i_invalid_time_keeps_the_keywords(self):
        params = _parse(
            {"keywords": ["garden"], "time": {"phrase": "last week", "unit": "fortnight"}},
            "garden last week",
        )
        assert params == {"keywords": ["garden"], "questions": []}

    def test_whole_query_quoted_as_time_is_dropped(self):
        # Observed on gemma4:e2b: for a request naming no time the model
        # quoted the entire query as the time phrase with unit "day".
        query = "recommend a restaurant I'd enjoy"
        assert _range({"phrase": "recommend a restaurant i'd enjoy", "unit": "day", "offset": 0},
                      query) == {}

    def test_overlong_quote_is_not_a_time_expression(self):
        query = "tell me what we said about the new kitchen layout plan yesterday evening"
        assert _range({"phrase": "what we said about the new kitchen layout plan", "unit": "day",
                       "offset": -1}, query) == {}

    def test_query_that_is_only_a_time_expression_still_resolves(self):
        assert _range({"phrase": "and yesterday", "unit": "day", "offset": -1},
                      "and yesterday?") != {}

    def test_quote_absent_from_query_is_dropped(self):
        assert _range({"phrase": "yesterday", "unit": "day", "offset": -1},
                      "what did we discuss about the garden?") == {}


class TestSafeFailure:
    def test_h_truncated_json_retries_once_then_returns_empty(self):
        truncated = '{"keywords": ["garden", "plants"], "questions": ["what'
        result, calls = _run("q", [truncated, truncated])
        assert result == {}
        assert len(calls) == 2

    def test_h_retry_repeats_the_identical_request(self):
        _, calls = _run("garden notes", ['{"keywords": ["gar', '{"keywords": ["garden"]}'])
        assert len(calls) == 2
        assert calls[0] == calls[1]

    def test_truncated_first_attempt_recovers_on_retry(self):
        result, calls = _run("q", ['{"keywords": ["gar', '{"keywords": ["garden"]}'])
        assert result["keywords"] == ["garden"]
        assert len(calls) == 2

    def test_backend_failure_retries_once(self):
        result, calls = _run("q", [None, None])
        assert result == {}
        assert len(calls) == 2

    @pytest.mark.parametrize("bad", [
        '{"keywords": "garden"}',
        '["garden"]',
        '{"questions": ["x"]}',
        '{"keywords": ["garden", 3]}',
        '{"keywords": ["garden"], "questions": "who?"}',
        '{"keywords": ["garden"], "questions": [null]}',
        '{"keywords": ["garden"], "time": "yesterday"}',
        'Sure! {"keywords": ["garden"]}',
        "not json at all",
    ])
    def test_schema_violations_are_rejected(self, bad):
        assert parse_search_params(bad, "q", now_local=NOW) is None

    def test_code_fenced_json_is_accepted(self):
        params = parse_search_params('```json\n{"keywords": ["garden"]}\n```', "q", now_local=NOW)
        assert params == {"keywords": ["garden"], "questions": []}

    def test_lists_are_cleaned_and_capped(self):
        params = parse_search_params(
            json.dumps({"keywords": ["garden", " ", "Garden", "x" * 200] + [f"k{i}" for i in range(20)],
                        "questions": ["who?"] * 3}),
            "q", now_local=NOW,
        )
        keywords_schema = SEARCH_PARAMS_SCHEMA["properties"]["keywords"]
        assert params["keywords"][0] == "garden"
        assert "x" * 200 not in params["keywords"]
        assert len(params["keywords"]) <= keywords_schema["maxItems"]
        assert params["questions"] == ["who?"]


class TestObservability:
    def test_debug_trace_covers_each_stage(self):
        messages = []
        with patch("jarvis.reply.enrichment.debug_log",
                   side_effect=lambda msg, cat="debug": messages.append(msg)):
            _run("what did I eat yesterday?", [json.dumps({
                "keywords": ["food"],
                "time": {"phrase": "yesterday", "unit": "day", "offset": -1},
            })])
        trace = "\n".join(messages)
        for stage in ("input", "raw", "validation", "time", "result"):
            assert f"search params {stage}" in trace, stage

    def test_nothing_is_printed_to_stdout(self, capsys):
        _run("my secret project codename", ['{"keywords": ["codename"]}'])
        assert "codename" not in capsys.readouterr().out


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
