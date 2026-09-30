"""Memory grounding: only stored facts relevant to the current message reach
the reply, relevant ones always do, and the selection is cheap, bounded and
observable.

The graph is a real SQLite ``GraphMemoryStore``; model calls are stubbed.
"""

import copy
import json
import threading
from unittest.mock import Mock, patch

import pytest

from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.graph import BRANCH_DIRECTIVES, BRANCH_USER, BRANCH_WORLD, GraphMemoryStore, named_terms
from jarvis.memory.graph_ops import STORED_FACTS_CLOSE, STORED_FACTS_OPEN
from src.jarvis.reply import enrichment
from src.jarvis.reply.engine import _grounding_result, run_reply_engine

pytestmark = pytest.mark.unit

FOOD = "The user loves Thai food, especially pad see ew"
RAMEN = "The user cooks homemade ramen every Sunday"
BOXING = "The user boxes three times a week at Trenches Gym"
PROJECT = "The user's project, NovaForge, uses PHP and MySQL"
CAT = "The user has a tabby cat called miso"
USER_FACTS = [FOOD, RAMEN, BOXING, PROJECT, CAT]
GYM_HOURS = "Trenches Gym opens at 6am"


def _seed(path, user_facts=USER_FACTS, world=GYM_HOURS, directive=None):
    store = GraphMemoryStore(path)
    if user_facts:
        store.update_node(BRANCH_USER, data="\n".join(user_facts))
    if world:
        store.update_node(BRANCH_WORLD, data=world)
    if directive:
        store.update_node(BRANCH_DIRECTIVES, data=directive)
    store.close()
    return path


@pytest.fixture
def db_path(tmp_path):
    return _seed(str(tmp_path / "jarvis.db"))


def _cfg(db_path):
    cfg = Mock()
    cfg.db_path = db_path
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "gemma4:e2b"
    cfg.llm_chat_model = "gemma4:e2b"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "all"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 4
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "llm"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.evaluator_enabled = False
    cfg.planner_enabled = True
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    cfg.llm_decision_temperature = 0.0
    return cfg


class _Run:
    def __init__(self, messages, plan, selector, reply):
        self.messages = messages
        self.plan = plan
        self.selector = selector
        self.reply = reply

    @property
    def user_turn(self):
        return self.messages[-1]["content"]

    @property
    def system(self):
        return self.messages[0]["content"]

    def block_lines(self):
        turn = self.user_turn
        if STORED_FACTS_OPEN not in turn:
            return []
        body = turn.split(STORED_FACTS_OPEN, 1)[1].split(STORED_FACTS_CLOSE, 1)[0]
        return [line for line in body.strip().splitlines() if line.strip()]


def _run(db_path, text, *, picks=(), tools=("stop",), plan=("Reply to the user.",),
         reply="Noted.", dialogue_memory=None, extract=None, selector=None):
    chats = []

    def fake_chat(**kwargs):
        chats.append(copy.deepcopy(kwargs["messages"]))
        return {"message": {"role": "assistant", "content": reply}}

    selector = selector or Mock(return_value=None if picks is None else list(picks))
    with patch("src.jarvis.reply.engine.plan_query", return_value=list(plan)) as plan_mock, \
         patch("src.jarvis.reply.engine.select_tools", return_value=list(tools)), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value=extract or {}), \
         patch("src.jarvis.reply.engine.select_relevant_facts", selector), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=fake_chat):
        out = run_reply_engine(db=Mock(), cfg=_cfg(db_path), tts=None, text=text,
                               dialogue_memory=dialogue_memory)
    return _Run(chats[0], plan_mock, selector, out)


# ── Named terms ─────────────────────────────────────────────────────────────


class TestNamedTerms:
    def test_capitalised_words_are_terms_but_sentence_case_is_not(self):
        assert named_terms(FOOD) == {"thai"}

    def test_names_with_inner_capitals_and_acronyms(self):
        assert named_terms(PROJECT) == {"novaforge", "php", "mysql"}

    def test_multi_word_name_at_the_start_of_a_line(self):
        assert named_terms(GYM_HOURS) == {"trenches", "gym"}

    def test_first_word_after_a_full_stop_is_sentence_case(self):
        assert named_terms("The user boxes. Friends call it a habit.") == frozenset()

    def test_lowercase_words_are_never_terms(self):
        assert named_terms(CAT) == frozenset()

    def test_scripts_without_letter_case_yield_no_terms(self):
        assert named_terms("用户喜欢泰国菜") == frozenset()


class TestWorldFactLines:
    def test_fact_on_the_world_branch_itself_is_found_by_named_term(self, db_path):
        store = GraphMemoryStore(db_path)
        hits = store.find_fact_lines_naming("what time does trenches gym open?", BRANCH_WORLD)
        store.close()
        assert [line for _, line in hits] == [GYM_HOURS]

    def test_unrelated_query_finds_nothing(self, db_path):
        store = GraphMemoryStore(db_path)
        assert store.find_fact_lines_naming("what's 17 times 23?", BRANCH_WORLD) == []
        store.close()

    def test_lines_naming_more_query_words_rank_first_and_limit_applies(self, tmp_path):
        path = _seed(str(tmp_path / "g.db"), user_facts=[],
                     world="Trenches Gym opens at 6am\nMorley Gym has a pool")
        store = GraphMemoryStore(path)
        hits = store.find_fact_lines_naming("is trenches gym open?", BRANCH_WORLD, limit=1)
        store.close()
        assert [line for _, line in hits] == ["Trenches Gym opens at 6am"]

    def test_excluded_nodes_are_skipped(self, db_path):
        store = GraphMemoryStore(db_path)
        assert store.find_fact_lines_naming(
            "trenches gym", BRANCH_WORLD, exclude_node_ids=frozenset({BRANCH_WORLD}),
        ) == []
        store.close()


# ── Selector contract ───────────────────────────────────────────────────────


class TestParseFactSelection:
    @pytest.mark.parametrize("response,expected", [
        ('{"facts": [2, 1]}', [0, 1]),
        ('{"facts": []}', []),
        ('{"facts": [3, 3]}', [2]),
        ('```json\n{"facts": [1]}\n```', [0]),
    ])
    def test_valid_answers(self, response, expected):
        assert enrichment.parse_fact_selection(response, 5) == expected

    @pytest.mark.parametrize("response", [
        "", "   ", "not json", '{"facts": [7]}', '{"facts": [0]}', '{"facts": [true]}',
        '{"facts": ["1"]}', '{"facts": 1}', '{"picks": [1]}', '[1, 2]',
        '{"facts": [1, 2, 3, 4, 5, 6]}', 'Sure: {"facts": [1]}',
    ])
    def test_invalid_answers_are_rejected(self, response):
        assert enrichment.parse_fact_selection(response, 6) is None


class TestSelectRelevantFacts:
    def _call(self, responses, facts=USER_FACTS, context_hint=None):
        mock = Mock(side_effect=list(responses))
        with patch.object(enrichment, "call_llm_direct", mock):
            result = enrichment.select_relevant_facts(
                "what should I cook?", facts, _cfg(":memory:"), "gemma4:e2b",
                context_hint=context_hint,
            )
        return result, mock

    def test_request_is_schema_bound_deterministic_and_static(self):
        result, mock = self._call(['{"facts": [1]}'], context_hint="Location: Disabled")
        assert result == [0]
        kwargs = mock.call_args.kwargs
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 32
        assert kwargs["system_prompt"] == enrichment._FACT_SELECTION_SYSTEM_PROMPT
        items = kwargs["json_schema"]["properties"]["facts"]
        assert items["items"]["enum"] == [1, 2, 3, 4, 5]
        assert items["maxItems"] == enrichment.FACT_SELECTION_MAX_PICKS
        content = kwargs["user_content"]
        assert f"1. {FOOD}" in content and f"5. {CAT}" in content
        assert "Location: Disabled" in content
        assert content.rstrip().endswith("Current message: what should I cook?")

    def test_invalid_answer_retries_once_with_the_same_request_then_gives_up(self):
        result, mock = self._call(['{"facts": [9]}', "oops"])
        assert result is None
        assert mock.call_count == 2
        assert mock.call_args_list[0].kwargs == mock.call_args_list[1].kwargs

    def test_recovers_on_the_retry(self):
        result, mock = self._call([None, '{"facts": [3]}'])
        assert result == [2]
        assert mock.call_count == 2

    def test_backend_error_gives_none(self):
        mock = Mock(side_effect=RuntimeError("boom"))
        with patch.object(enrichment, "call_llm_direct", mock):
            assert enrichment.select_relevant_facts("q", USER_FACTS, _cfg(":memory:"), "m") is None

    def test_no_facts_means_no_call(self):
        result, mock = self._call([], facts=[])
        assert result == []
        mock.assert_not_called()

    def test_no_model_means_no_call(self):
        mock = Mock()
        with patch.object(enrichment, "call_llm_direct", mock):
            assert enrichment.select_relevant_facts("q", USER_FACTS, _cfg(":memory:"), "") is None
        mock.assert_not_called()

    def test_candidates_are_capped(self):
        facts = [f"Fact number {i}" for i in range(60)]
        _, mock = self._call(['{"facts": []}'], facts=facts)
        schema = mock.call_args.kwargs["json_schema"]
        assert schema["properties"]["facts"]["items"]["enum"][-1] == enrichment.FACT_SELECTION_MAX_CANDIDATES
        assert "Fact number 40" not in mock.call_args.kwargs["user_content"]


# ── Engine grounding (Phase 8 regression set) ───────────────────────────────


class TestEngineGrounding:
    def test_1_direct_personal_fact_reaches_the_reply(self, db_path):
        run = _run(db_path, "what do you know about my food preferences?", picks=[0])
        assert run.block_lines() == [FOOD]
        assert run.user_turn.endswith("CURRENT USER MESSAGE:\nwhat do you know about my food preferences?")

    def test_2_open_ended_prompt_gets_facts_to_build_on(self, db_path):
        run = _run(db_path, "say something", picks=[0, 4])
        assert run.block_lines() == [FOOD, CAT]

    def test_3_4_relevant_fact_in_and_unrelated_facts_out(self, db_path):
        run = _run(db_path, "any tips for my boxing this week?", picks=[2])
        assert run.block_lines() == [BOXING]
        for other in (FOOD, RAMEN, PROJECT, CAT):
            assert other not in run.user_turn and other not in run.system

    def test_4_unrelated_query_gets_no_stored_facts_at_all(self, db_path):
        run = _run(db_path, "what is the weather tomorrow?", picks=[], tools=("getWeather", "stop"))
        assert run.user_turn == "what is the weather tomorrow?"
        assert all(f not in run.system for f in USER_FACTS)
        assert run.plan.call_args.kwargs["stored_facts"] == ""

    def test_5_named_personal_fact_is_included_even_if_the_model_picks_nothing(self, db_path):
        run = _run(db_path, "what stack is NovaForge built on?", picks=[], tools=("webSearch", "stop"))
        assert run.block_lines() == [PROJECT]
        assert PROJECT in run.plan.call_args.kwargs["stored_facts"]
        assert "take precedence over web or tool results" in run.user_turn

    def test_5_every_fact_named_means_no_selection_call(self, tmp_path):
        path = _seed(str(tmp_path / "named.db"), user_facts=[PROJECT, FOOD], world=None)
        run = _run(path, "is NovaForge still on PHP, and is Thai food still my favourite?", picks=[])
        run.selector.assert_not_called()
        assert run.block_lines() == [PROJECT, FOOD]

    def test_6_current_web_questions_keep_their_tools(self, db_path):
        run = _run(db_path, "what's the latest PHP news?", picks=[], tools=("webSearch", "stop"),
                   plan=("webSearch query='PHP news'", "Reply to the user."))
        tool_block = run.system
        assert "webSearch" in tool_block
        assert PROJECT in run.plan.call_args.kwargs["stored_facts"]

    def test_7_changing_information_prefers_this_turns_tool_result_and_states_conflicts(self, db_path):
        run = _run(db_path, "what time does Trenches Gym open today?", picks=[2], tools=("webSearch", "stop"))
        lines = run.block_lines()
        assert BOXING in lines
        assert any(line.endswith(GYM_HOURS) for line in lines)
        assert "a tool result from this turn is newer: prefer it" in run.user_turn
        assert "say so briefly instead of merging the two" in run.user_turn

    def test_8_multiple_relevant_facts_keep_their_stored_order(self, db_path):
        run = _run(db_path, "what should I eat on Sunday?", picks=[1, 0])
        assert run.block_lines() == [FOOD, RAMEN]

    def test_9_memory_survives_a_restart(self, tmp_path):
        path = str(tmp_path / "restart.db")
        store = GraphMemoryStore(path)
        store.update_node(BRANCH_USER, data=FOOD)
        store.close()
        run = _run(path, "what do I like to eat?", picks=[0])
        assert run.block_lines() == [FOOD]

    def test_10_no_memory_means_a_plain_turn_and_no_selection_call(self, tmp_path):
        path = _seed(str(tmp_path / "empty.db"), user_facts=[], world=None)
        run = _run(path, "tell me a joke", picks=[0])
        run.selector.assert_not_called()
        assert run.user_turn == "tell me a joke"


class TestSelectionMechanics:
    def test_selection_failure_falls_back_to_every_warm_fact(self, db_path, capsys):
        run = _run(db_path, "hello", picks=None)
        assert run.block_lines() == USER_FACTS
        assert "selection unavailable, using all" in capsys.readouterr().out

    def test_selection_overlaps_the_tool_router(self, db_path):
        started = threading.Event()
        seen_by_router = []

        def slow_selector(*args, **kwargs):
            started.set()
            return [0]

        def router(**kwargs):
            seen_by_router.append(started.wait(timeout=5))
            return ["stop"]

        with patch("src.jarvis.reply.engine.select_tools", side_effect=router), \
             patch("src.jarvis.reply.engine.plan_query", return_value=["Reply to the user."]), \
             patch("src.jarvis.reply.engine.select_relevant_facts", side_effect=slow_selector), \
             patch("src.jarvis.reply.engine.chat_with_messages",
                   return_value={"message": {"role": "assistant", "content": "ok"}}):
            run_reply_engine(db=Mock(), cfg=_cfg(db_path), tts=None, text="what should I cook?",
                             dialogue_memory=None)
        assert seen_by_router == [True]

    def test_selection_is_cached_for_the_same_query_in_one_conversation(self, db_path):
        dm = DialogueMemory(inactivity_timeout=300, max_interactions=20)
        selector = Mock(return_value=[0])
        _run(db_path, "what should I cook?", selector=selector, dialogue_memory=dm)
        second = _run(db_path, "what should I cook?", selector=selector, dialogue_memory=dm)
        assert selector.call_count == 1
        assert second.block_lines() == [FOOD]

    def test_selector_gets_the_query_warm_facts_and_live_context(self, db_path):
        run = _run(db_path, "how's the weather?", picks=[])
        args, kwargs = run.selector.call_args
        assert args[0] == "how's the weather?"
        assert args[1] == USER_FACTS
        assert "Location: Disabled" in kwargs["context_hint"]

    def test_graph_search_does_not_bring_back_whole_user_nodes(self, db_path):
        run = _run(db_path, "tell me what food I like", picks=[], plan=(),
                   extract={"keywords": ["food"], "questions": ["what food does the user like?"]})
        assert FOOD not in run.system
        assert FOOD not in run.user_turn


class TestObservability:
    def test_normal_output_shows_counts_not_fact_text(self, db_path, capsys):
        _run(db_path, "what should I cook?", picks=[0])
        out = capsys.readouterr().out
        assert "Stored facts: 1 of 5 relevant" in out
        assert "Thai" not in out and "pad see ew" not in out

    def test_debug_trace_covers_every_stage(self, db_path, capsys):
        with patch("src.jarvis.debug._is_debug_enabled", return_value=True):
            _run(db_path, "what should I cook?", picks=[0], reply="Perhaps some Thai food tonight.")
        err = capsys.readouterr().err
        for stage in ("grounding: query", "grounding: memory relevance yes",
                      "grounding: retrieved memory 1. [selected]", "grounding: prompt sections",
                      "grounding: memory section", "grounding: model response",
                      "grounding: result grounded"):
            assert stage in err, stage

    def test_debug_trace_is_silent_when_debug_is_off(self, db_path, capsys):
        _run(db_path, "what should I cook?", picks=[0])
        assert "grounding:" not in capsys.readouterr().err


class TestGroundingResult:
    def test_reply_using_a_distinctive_word_is_grounded(self):
        assert _grounding_result("Your NovaForge stack is PHP.", [PROJECT], USER_FACTS, "what stack?").startswith("grounded")

    def test_shared_boilerplate_and_query_words_do_not_count(self):
        verdict = _grounding_result("The user asked about cooking.", [FOOD], USER_FACTS, "what should I cook?")
        assert verdict.startswith("ungrounded")

    def test_no_facts_is_not_applicable(self):
        assert _grounding_result("Paris.", [], USER_FACTS, "capital of France?").startswith("not applicable")
