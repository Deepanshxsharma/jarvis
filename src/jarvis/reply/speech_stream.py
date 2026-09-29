"""Speak a reply sentence by sentence while the model is still writing it.

The chat model's content streams in token by token. :class:`SpeechStreamer`
decides early whether a turn is prose meant for the user or structured
output (a text-protocol tool call, JSON), and hands each finished prose
sentence to a speech sink. Structured turns are never spoken.

Contract with the reply engine:

- ``begin_turn()`` before each chat-model call, then ``feed(piece)`` for
  every content piece of that call.
- ``finish(final_reply)`` once the reply is decided. When the final reply
  is the turn that was streaming, only its unspoken tail is spoken. Any
  other final reply (an error message, a digest, a malformed-output
  fallback, an answer after a spoken preamble and a tool call) is spoken
  in full.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

from ..debug import debug_log

# A sentence ends at terminal punctuation (Latin, Devanagari danda, CJK
# full stops) followed by whitespace. A digit before the stop is not an
# ending, so list markers ("1. ") and decimals stay attached.
_SENTENCE_END = re.compile(r"(?<=[^\d\s][.!?\u0964\u0965\u3002\uff01\uff1f])\s+")

# Openings that mark a turn as structured output rather than speech.
_STRUCTURED_OPENINGS = ("tool_calls", "```", "{", "[", "<", "tool_code", "print(")

# Markers that, appearing later in a prose turn, mean the model switched to
# a tool call mid-turn; nothing after them may be spoken.
_TOOL_MARKERS = ("tool_calls", "```")

# Characters needed before deciding a turn is prose when no sentence has
# ended yet. Long enough to see a structured opening, short enough not to
# delay speech.
_DECIDE_AFTER_CHARS = 12


class SpeechStreamer:
    def __init__(self, sink: Callable[[str], None]) -> None:
        self._sink = sink
        self._turn_text = ""
        self._consumed = 0
        self._state = "undecided"
        self.spoken: List[str] = []

    def begin_turn(self) -> None:
        self._turn_text = ""
        self._consumed = 0
        self._state = "undecided"

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self._turn_text += piece
        if self._state == "held":
            return
        if self._state == "undecided":
            head = self._turn_text.lstrip()
            if not head:
                return
            lowered = head.lower()
            if lowered.startswith(_STRUCTURED_OPENINGS):
                self._hold("structured opening")
                return
            if len(head) < _DECIDE_AFTER_CHARS and not _SENTENCE_END.search(head + " "):
                return
            self._state = "speaking"
        lowered = self._turn_text.lower()
        marker_at = min((lowered.find(m) for m in _TOOL_MARKERS if m in lowered), default=-1)
        if marker_at >= 0:
            self._emit_complete_sentences(end=marker_at)
            self._hold("tool call inside prose")
            return
        self._emit_complete_sentences()

    def finish(self, final_reply: str) -> None:
        """Speak the part of ``final_reply`` that has not been spoken yet."""
        final = (final_reply or "").strip()
        if not final:
            return
        spoken_prefix = ""
        if self._state == "speaking":
            spoken_prefix = self._turn_text[:self._consumed].strip()
        if spoken_prefix and final.startswith(spoken_prefix):
            remainder = final[len(spoken_prefix):].strip()
        else:
            if spoken_prefix:
                debug_log("speech stream: final reply differs from the spoken turn", "voice")
            remainder = final
        if remainder:
            self._speak(remainder)

    def _hold(self, reason: str) -> None:
        self._state = "held"
        debug_log(f"speech stream: holding turn ({reason})", "voice")

    def _emit_complete_sentences(self, end: Optional[int] = None) -> None:
        pending = self._turn_text[self._consumed:end]
        last_end = 0
        for match in _SENTENCE_END.finditer(pending):
            sentence = pending[last_end:match.start()].strip()
            last_end = match.end()
            if sentence:
                self._speak(sentence)
        self._consumed += last_end

    def _speak(self, text: str) -> None:
        self.spoken.append(text)
        try:
            self._sink(text)
        except Exception as exc:
            debug_log(f"speech stream sink failed: {exc}", "voice")
