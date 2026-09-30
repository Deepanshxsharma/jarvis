"""Engine wiring for the memory search-parameter extractor: it receives the
user's local time for resolving relative dates, and the query-derived
keywords stay out of the normal console log."""

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine


def _mock_cfg():
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"
    cfg.llm_chat_model = "test-large"
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
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    cfg.db_path = ":memory:"
    return cfg


@pytest.mark.unit
@patch("src.jarvis.memory.graph_ops.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.graph_ops.build_warm_profile", return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.select_tools", return_value=["webSearch"])
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory",
       return_value={"keywords": ["zanzibarcodename"], "questions": []})
@patch("src.jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[])
@patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok")
@patch("src.jarvis.reply.engine.chat_with_messages", return_value={"message": {"content": "ok"}})
def test_extractor_gets_local_time_and_keywords_stay_out_of_console(
    _mock_chat, _mock_extract, _mock_search, mock_extractor,
    _mock_plan, _mock_select, _mock_graph, _mock_warm, _mock_fmt, capsys,
):
    run_reply_engine(db=Mock(), cfg=_mock_cfg(), tts=None,
                     text="what did I say yesterday about the trip",
                     dialogue_memory=DialogueMemory())

    assert mock_extractor.call_count == 1
    now_local = mock_extractor.call_args.kwargs.get("now_local")
    assert now_local is not None and now_local.utcoffset() is not None

    out = capsys.readouterr().out
    assert "Memory search" in out
    assert "zanzibarcodename" not in out
