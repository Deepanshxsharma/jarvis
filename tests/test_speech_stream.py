"""Replies are spoken sentence by sentence while the model writes them."""

from jarvis.reply.speech_stream import SpeechStreamer


def _stream(pieces, final=None):
    spoken = []
    streamer = SpeechStreamer(spoken.append)
    streamer.begin_turn()
    for piece in pieces:
        streamer.feed(piece)
    during = list(spoken)
    streamer.finish(final if final is not None else "".join(pieces))
    return during, spoken


class TestSentenceStreaming:
    def test_each_sentence_is_spoken_as_soon_as_it_ends(self):
        spoken = []
        streamer = SpeechStreamer(spoken.append)
        streamer.begin_turn()
        for piece in ["It is sun", "ny in Delhi", ". Tomorrow will ", "be cooler"]:
            streamer.feed(piece)
        assert spoken == ["It is sunny in Delhi."]
        streamer.feed(". Take an umbrella.")
        assert spoken == ["It is sunny in Delhi.", "Tomorrow will be cooler."]

    def test_finish_speaks_only_the_unspoken_tail(self):
        pieces = ["Paris is the capital. ", "It is in France."]
        during, spoken = _stream(pieces)
        assert during == ["Paris is the capital."]
        assert spoken == ["Paris is the capital.", "It is in France."]

    def test_numbered_items_and_decimals_do_not_end_a_sentence(self):
        during, spoken = _stream(["Steps: 1. open it. It costs 2.5 dollars. Done"])
        assert spoken == ["Steps: 1. open it.", "It costs 2.5 dollars.", "Done"]

    def test_devanagari_danda_ends_a_sentence(self):
        during, _ = _stream(["आज मौसम अच्छा है। ", "कल बारिश होगी।"])
        assert during == ["आज मौसम अच्छा है।"]

    def test_short_reply_is_spoken_once_on_finish(self):
        during, spoken = _stream(["Sure"])
        assert during == []
        assert spoken == ["Sure"]


class TestStructuredOutputIsNeverSpoken:
    def test_text_tool_call_turn_is_held(self):
        spoken = []
        streamer = SpeechStreamer(spoken.append)
        streamer.begin_turn()
        for piece in ["tool_calls: ", '[{"name": "webSearch", ', '"arguments": {}}]. ', "More. "]:
            streamer.feed(piece)
        assert spoken == []

    def test_json_turn_is_held(self):
        during, _ = _stream(['{"response": "Hi there. ', 'How are you?"}'], final="Hi there. How are you?")
        assert during == []

    def test_tool_call_after_prose_stops_speech_at_the_marker(self):
        spoken = []
        streamer = SpeechStreamer(spoken.append)
        streamer.begin_turn()
        for piece in ["Let me look that up. ", "tool_calls: [{", '"name": "x"}]. Then more. ']:
            streamer.feed(piece)
        assert spoken == ["Let me look that up."]

    def test_answer_after_a_tool_turn_is_spoken_in_full(self):
        spoken = []
        streamer = SpeechStreamer(spoken.append)
        streamer.begin_turn()
        streamer.feed("Let me look that up. tool_calls: [...]")
        streamer.begin_turn()
        streamer.feed("It is 30 degrees. ")
        streamer.feed("Quite warm.")
        streamer.finish("It is 30 degrees. Quite warm.")
        assert spoken == ["Let me look that up.", "It is 30 degrees.", "Quite warm."]

    def test_final_reply_from_elsewhere_is_spoken_whole(self):
        spoken = []
        streamer = SpeechStreamer(spoken.append)
        streamer.begin_turn()
        streamer.feed("tool_calls: [...]")
        streamer.finish("Sorry, I had trouble with that.")
        assert spoken == ["Sorry, I had trouble with that."]


def test_a_failing_sink_does_not_break_the_stream():
    calls = []

    def sink(text):
        calls.append(text)
        raise RuntimeError("audio gone")

    streamer = SpeechStreamer(sink)
    streamer.begin_turn()
    streamer.feed("One sentence here. Two")
    streamer.finish("One sentence here. Two")
    assert calls == ["One sentence here.", "Two"]
