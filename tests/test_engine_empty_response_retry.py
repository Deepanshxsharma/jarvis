"""An empty chat reply on the text-tools path is retried exactly once with the
allowed tools declared natively.

Small gemma models sometimes emit their native tool-call tokens even when no
tools are declared; the server then returns empty content and the reply used
to end in a generic apology. Declaring the tools makes the same call surface
as structured ``tool_calls``.
"""

import copy
from unittest.mock import Mock, patch

import pytest

from src.jarvis.reply.engine import run_reply_engine

pytestmark = pytest.mark.unit


def _cfg(model):
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = model
    cfg.llm_chat_model = model
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.evaluator_enabled = False
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    return cfg


EMPTY = {"message": {"role": "assistant", "content": ""}}
NATIVE_CALL = {"message": {"role": "assistant", "content": "", "tool_calls": [{
    "function": {"name": "webSearch", "arguments": {"search_query": "latest tech news"}},
}]}}
FINAL = {"message": {"role": "assistant", "content": "Here are today's tech headlines."}}


def _run(model, responses):
    calls = []

    def fake_chat(**kwargs):
        calls.append({
            "tools": kwargs.get("tools"),
            "messages": copy.deepcopy(kwargs.get("messages")),
        })
        return responses[len(calls) - 1] if len(calls) <= len(responses) else FINAL

    tool_result = Mock(reply_text="Headline: a new chip was announced.", error_message=None)
    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.run_tool_with_retries", return_value=tool_result) as tool, \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=fake_chat):
        reply = run_reply_engine(
            db=Mock(), cfg=_cfg(model), tts=None,
            text="what's happening in tech news?", dialogue_memory=None,
        )
    return reply, calls, tool


def _tool_names(schema):
    return {t["function"]["name"] for t in schema or []}


def test_empty_reply_is_retried_once_with_tool_declarations(capsys):
    reply, calls, tool = _run("gemma4:e2b", [EMPTY, NATIVE_CALL, FINAL])

    assert calls[0]["tools"] is None, "text-tools path sends no native tools on the first try"
    assert "webSearch" in _tool_names(calls[1]["tools"])
    # Same request and context: the empty assistant turn is not fed back.
    assert calls[1]["messages"] == calls[0]["messages"]
    assert calls[1]["messages"][-1] == {"role": "user", "content": "what's happening in tech news?"}

    assert tool.call_args.kwargs["tool_name"] == "webSearch"
    assert tool.call_args.kwargs["tool_args"] == {"search_query": "latest tech news"}
    assert reply == "Here are today's tech headlines."
    assert "retrying once" in capsys.readouterr().out


def test_retry_happens_at_most_once_per_reply():
    reply, calls, tool = _run("gemma4:e2b", [EMPTY, EMPTY, EMPTY, EMPTY])

    with_tools = [c for c in calls if c["tools"]]
    assert len(with_tools) == 1
    assert len(calls) == 2
    tool.assert_not_called()
    assert reply and "trouble" in reply.lower()


def test_native_tools_path_is_not_retried():
    _, calls, _ = _run("test-large", [EMPTY, EMPTY])

    assert len(calls) == 1
