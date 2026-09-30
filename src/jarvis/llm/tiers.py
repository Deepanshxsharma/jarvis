"""The two-tier model system.

Jarvis runs every LLM context on one of two models:

- ``Tier.FAST`` — the small, warm, low-latency model behind real-time
  classification passes. These contexts take a few thousand tokens in and
  emit tiny strict-JSON answers, so latency dominates and a ~2B model is
  ideal.
- ``Tier.CHAT`` — the capable model that writes replies, plans,
  summarises, and extracts knowledge. Long-form output; quality dominates.

The Model tiers table in ``llm.spec.md`` is the authoritative list of
which context runs on which tier; docstrings elsewhere point here rather
than re-enumerating it.

Both fields are fully resolved at config load (``fast_model`` /
``llm_chat_model`` always hold a provider-valid model name), so resolution
here is a plain field read. Keeping it behind one function means every
context states its tier instead of inventing a fallback chain, and any
future routing logic lands in exactly one place.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Which of the two models a context runs on."""

    FAST = "fast"
    CHAT = "chat"


def resolve_model(cfg, tier: Tier) -> str:
    """Return the model name for ``tier`` under the active settings.

    ``Tier.FAST`` reads ``cfg.fast_model``; ``Tier.CHAT`` reads
    ``cfg.llm_chat_model``. An empty fast model (possible only on
    hand-built cfg objects — config load always resolves it) falls back
    to the chat model so a context never receives an empty name.
    """
    chat = str(getattr(cfg, "llm_chat_model", "") or "").strip()
    if tier is Tier.FAST:
        return str(getattr(cfg, "fast_model", "") or "").strip() or chat
    return chat


DEFAULT_DECISION_TEMPERATURE = 0.0


def decision_temperature(cfg) -> float:
    """Sampling temperature for decision contexts (tool routing, task
    planning, step resolution, memory-search extraction).

    These contexts pick one structured answer rather than write prose, so
    sampling variance only adds flakiness: at the model default (1.0 for
    gemma4) the same query can route or plan differently run to run.
    Reads ``cfg.llm_decision_temperature``; hand-built cfg objects without
    the field, or with a non-numeric / negative value, get the default.
    """
    raw = getattr(cfg, "llm_decision_temperature", DEFAULT_DECISION_TEMPERATURE)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_DECISION_TEMPERATURE
    if value < 0:
        return DEFAULT_DECISION_TEMPERATURE
    return value
