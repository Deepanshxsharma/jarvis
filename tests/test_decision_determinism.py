"""Decision contexts (router, planner, step resolver) sample at the configured
decision temperature, and the router reads recent dialogue as background.

At the model's default temperature the same query could route or plan
differently run to run, which made follow-up behaviour unreproducible.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import jarvis.reply.planner as planner_mod
from jarvis.reply.planner import plan_query, resolve_next_tool_call
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.tool_search import ToolSearchTool
from jarvis.tools.selection import ToolSelectionStrategy, select_tools

pytestmark = pytest.mark.unit


def _cfg(**over):
    base = dict(
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="gemma4:e2b",
        llm_chat_model="gemma4:e2b",
        fast_model="",
        planner_enabled=True,
        planner_timeout_sec=3.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _builtin():
    def tool(desc):
        return SimpleNamespace(description=desc)
    return {
        "webSearch": tool("Search the web for current information."),
        "getWeather": tool("Get the current weather and forecast."),
        "stop": tool("Stop the conversation."),
        "toolSearchTool": tool("Search for more tools."),
    }


def _router_capture(**select_kwargs):
    captured = {}

    def _direct(model, sys, user, timeout_sec=8.0, **kwargs):
        captured.update(sys=sys, user=user, kwargs=kwargs)
        return "webSearch"

    backend = MagicMock()
    backend.direct.side_effect = _direct
    select_tools(
        "is the bakery on the corner open late?",
        _builtin(), {},
        strategy=ToolSelectionStrategy.LLM,
        llm_backend=backend,
        llm_model="test",
        **select_kwargs,
    )
    return captured


class TestRouterSampling:
    def test_router_forwards_temperature(self):
        captured = _router_capture(llm_temperature=0.0)
        assert captured["kwargs"]["temperature"] == 0.0

    def test_router_defaults_to_model_sampling_when_unset(self):
        captured = _router_capture()
        assert captured["kwargs"]["temperature"] is None

    def test_tool_search_uses_decision_temperature(self, mock_config):
        mock_config.llm_decision_temperature = 0.0
        ctx = ToolContext(
            db=None, cfg=mock_config, system_prompt="", original_prompt="",
            redacted_text="", max_retries=0, user_print=lambda _m: None,
            language=None,
        )
        with patch(
            "jarvis.tools.builtin.tool_search.select_tools", return_value=["webSearch"],
        ) as sel:
            ToolSearchTool().run({"query": "find a fact"}, ctx)
        assert sel.call_args.kwargs["llm_temperature"] == 0.0


class TestRouterDialogueFraming:
    def test_dialogue_is_labelled_as_background_not_the_request(self):
        hint = (
            "Current local time: Wednesday 12:00.\n\n"
            "Recent dialogue (short-term memory):\n"
            "- user: how's the weather?\n"
            "- assistant: Overcast and 8 degrees."
        )
        user = _router_capture(context_hint=hint)["user"]
        label_line = next(line for line in user.splitlines() if line.startswith("RECENT DIALOGUE"))
        assert "background" in label_line.lower()
        assert "not the current request" in label_line.lower()
        # The current query still arrives after the dialogue as the thing to route.
        assert user.index("RECENT DIALOGUE") < user.index("User query: is the bakery")


class TestPlannerSampling:
    def test_plan_query_uses_decision_temperature(self):
        with patch.object(planner_mod, "call_llm_direct", return_value="Reply to user.") as spy:
            plan_query(_cfg(), "tell me something", "", [("webSearch", "Search the web")])
        assert spy.call_args.kwargs["temperature"] == 0.0

    def test_plan_query_honours_configured_temperature(self):
        with patch.object(planner_mod, "call_llm_direct", return_value="Reply to user.") as spy:
            plan_query(_cfg(llm_decision_temperature=0.5), "tell me something", "",
                       [("webSearch", "Search the web")])
        assert spy.call_args.kwargs["temperature"] == 0.5

    def test_step_resolver_uses_decision_temperature(self):
        schema = [{
            "type": "function",
            "function": {
                "name": "webSearch",
                "description": "Search the web.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]
        with patch.object(planner_mod, "call_llm_direct", return_value="null") as spy:
            resolve_next_tool_call(_cfg(), "search for <the film from step 1>", [], schema)
        assert spy.called
        assert spy.call_args.kwargs["temperature"] == 0.0
