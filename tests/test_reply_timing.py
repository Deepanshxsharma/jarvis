"""Per-reply stage timing: every reply reports where its time went."""

from unittest.mock import patch

import pytest

from jarvis.reply import timing


@pytest.mark.unit
class TestReplyTiming:
    def test_stages_accumulate_and_count_repeat_calls(self):
        record = timing.begin()
        with timing.stage("llm"):
            pass
        with timing.stage("llm"):
            pass
        with timing.stage("router"):
            pass

        assert record.calls == {"llm": 2, "router": 1}
        line = record.format()
        assert line.startswith("REPLY router=")
        assert "llm=" in line and "x2" in line
        assert line.rstrip().split()[-1].startswith("total=")

    def test_stage_outside_a_reply_is_harmless(self):
        timing._current.set(None)
        with timing.stage("llm"):
            pass

    def test_token_usage_comes_from_the_model_response(self):
        record = timing.begin()
        record.record_llm_usage({"prompt_eval_count": 4900, "eval_count": 30})
        record.record_llm_usage({"message": {"content": "no usage fields"}})
        assert "tokens=4900in/30out" in record.format()

    def test_stage_is_recorded_when_the_work_raises(self):
        record = timing.begin()
        with pytest.raises(RuntimeError):
            with timing.stage("tools"):
                raise RuntimeError("tool failed")
        assert record.calls["tools"] == 1


@pytest.mark.unit
def test_reply_engine_reports_timing_even_when_the_reply_fails(capsys):
    from jarvis.reply import engine

    with patch.object(engine, "_run_reply_engine_body", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            engine.run_reply_engine(None, object(), None, "hello", None)

    assert "⏱️ REPLY" in capsys.readouterr().out
    assert timing.last_reply_timing().finished is not None
