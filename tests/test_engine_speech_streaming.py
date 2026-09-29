"""The reply engine hands speech out sentence by sentence and can be cancelled.

Uses a mocked chat model that streams its content through ``on_text``.
"""

import threading
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _hermetic():
    with patch("jarvis.reply.engine.plan_query", return_value=[]), \
         patch("jarvis.reply.engine.extract_search_params_for_memory", return_value={"keywords": []}):
        yield


def _streaming_chat(turns, on_call=None):
    """A fake chat call that streams each turn's pieces then returns them."""
    calls = []

    def chat(*_args, on_text=None, cancel=None, **_kwargs):
        pieces, tool_calls = turns[min(len(calls), len(turns) - 1)]
        calls.append(pieces)
        for piece in pieces:
            if cancel is not None and cancel.is_set():
                return None
            if on_text is not None:
                on_text(piece)
            if on_call is not None:
                on_call(len(calls), piece)
        if cancel is not None and cancel.is_set():
            return None
        message = {"role": "assistant", "content": "".join(pieces)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"message": message}

    return chat, calls


@pytest.mark.unit
def test_sentences_are_handed_out_while_the_reply_is_written(mock_config, db, dialogue_memory):
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_chat_model = "gemma4:e2b"
    spoken, seen_during_generation = [], []
    chat, _ = _streaming_chat(
        [(["Paris is the capital", " of France. It sits", " on the Seine."], None)],
        on_call=lambda _n, _p: seen_during_generation.append(list(spoken)),
    )

    with patch("jarvis.reply.engine.chat_with_messages", side_effect=chat):
        reply = run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="capital of france",
            dialogue_memory=dialogue_memory, on_speech=spoken.append,
        )

    assert reply == "Paris is the capital of France. It sits on the Seine."
    assert ["Paris is the capital of France."] in seen_during_generation
    assert spoken == ["Paris is the capital of France.", "It sits on the Seine."]


@pytest.mark.unit
def test_tool_call_turns_are_not_spoken(mock_config, db, dialogue_memory):
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    spoken = []
    call = {"id": "c1", "function": {"name": "getWeather", "arguments": {"location": "Delhi"}}}
    chat, calls = _streaming_chat([([""], [call]), (["It is 30 degrees in Delhi."], None)])

    with patch("jarvis.reply.engine.chat_with_messages", side_effect=chat), \
         patch("jarvis.reply.engine.select_tools", return_value=["getWeather", "stop"]), \
         patch("jarvis.reply.engine.run_tool_with_retries",
               return_value=ToolExecutionResult(success=True, reply_text="Delhi: 30C")):
        reply = run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="weather in delhi",
            dialogue_memory=dialogue_memory, on_speech=spoken.append,
        )

    assert len(calls) == 2
    assert "30 degrees" in reply
    assert " ".join(spoken) == reply


@pytest.mark.unit
def test_cancelling_stops_generation_and_returns_nothing(mock_config, db, dialogue_memory):
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_chat_model = "gemma4:e2b"
    cancel = threading.Event()
    spoken = []

    def on_speech(sentence):
        spoken.append(sentence)
        cancel.set()

    chat, calls = _streaming_chat([(["First sentence here. ", "Second sentence. ", "Third."], None)])

    with patch("jarvis.reply.engine.chat_with_messages", side_effect=chat):
        reply = run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="tell me three things",
            dialogue_memory=dialogue_memory, on_speech=on_speech, cancel_event=cancel,
        )

    assert reply is None
    assert spoken == ["First sentence here."]
    assert len(calls) == 1


@pytest.mark.unit
def test_cancel_before_the_first_turn_skips_the_model(mock_config, db, dialogue_memory):
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_chat_model = "gemma4:e2b"
    cancel = threading.Event()
    cancel.set()
    chat, calls = _streaming_chat([(["Hello."], None)])

    with patch("jarvis.reply.engine.chat_with_messages", side_effect=chat):
        reply = run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="hello",
            dialogue_memory=dialogue_memory, on_speech=lambda _s: None, cancel_event=cancel,
        )

    assert reply is None
    assert calls == []
