"""Native-rate microphone frames reach VAD without silent rejection or loss."""

from collections import deque
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener

pytestmark = pytest.mark.unit


def listener(rate=48000):
    obj = VoiceListener.__new__(VoiceListener)
    obj.cfg = SimpleNamespace(voice_min_energy=0.0045, voice_debug=False)
    obj._samplerate = 16000
    obj._stream_samplerate = rate
    obj._frame_samples = rate * 20 // 1000
    obj._pending_audio = None
    obj._recent_audio_energy = deque(maxlen=20)
    obj._vad_error_logged = False
    obj._vad = None
    obj._should_stop = obj._dictation_active = False
    obj._callback_count = 0
    obj._audio_q = queue.Queue(maxsize=2)
    obj._reset_audio_health(now=0)
    return obj


@pytest.mark.parametrize('rate', [16000, 44100, 48000])
def test_native_frames_use_supported_vad_format(rate):
    obj = listener(rate)

    class StrictVad:
        def is_speech(self, pcm, sample_rate):
            assert sample_rate == 16000
            assert len(pcm) == 640  # 20 ms, mono int16 at 16 kHz
            return True

    obj._vad = StrictVad()
    assert obj._is_speech_frame(np.ones(obj._frame_samples, dtype=np.float32) * .1)


def test_partial_callback_frames_are_not_discarded():
    obj = listener(44100)
    audio = np.linspace(-.1, .1, obj._frame_samples * 3 + 17, dtype=np.float32)
    output = []
    for block in np.array_split(audio, 13):
        output.extend(obj._audio_frames(block[:, None]))
    np.testing.assert_array_equal(np.concatenate(output), audio[:-17])
    np.testing.assert_array_equal(obj._pending_audio, audio[-17:])


def test_vad_failure_warns_once_and_uses_energy_gate(capsys):
    obj = listener()
    class BrokenVad:
        def is_speech(self, *args):
            raise ValueError('invalid frame')
    obj._vad = BrokenVad()
    assert obj._is_speech_frame(np.ones(obj._frame_samples) * .1)
    assert not obj._is_speech_frame(np.zeros(obj._frame_samples))
    assert capsys.readouterr().out.count('Speech detection failed') == 1


def test_capture_health_distinguishes_missing_callbacks_and_silent_samples(capsys):
    obj = listener()
    obj._check_audio_health(now=6)
    assert 'No microphone callbacks' in capsys.readouterr().out
    obj._callback_count = 1
    obj._last_audio_callback = 11
    obj._audio_frames(np.zeros((obj._frame_samples, 1), dtype=np.float32))
    obj._check_audio_health(now=12)
    assert 'silent samples' in capsys.readouterr().out


def test_callback_status_and_queue_overflow_are_visible(capsys):
    obj = listener()
    for _ in range(3):
        obj._on_audio(np.ones((960, 1), dtype=np.float32), 960, None, 'input overflow')
    obj._check_audio_health(now=6)
    output = capsys.readouterr().out
    assert 'input overflow' in output
    assert '1' in output and 'dropped' in output


def test_dictation_pause_does_not_report_capture_failure(capsys):
    obj = listener()
    obj._dictation_active = True
    obj._check_audio_health(now=30)
    assert not capsys.readouterr().out


def test_stalled_capture_warns_once_then_reports_recovery(capsys):
    obj = listener()
    obj._last_audio_callback = 2
    obj._check_audio_health(now=8)
    obj._check_audio_health(now=14)
    assert capsys.readouterr().out.count('No microphone callbacks') == 1
    obj._last_audio_callback = 19
    obj._audio_frames(np.ones((obj._frame_samples, 1), dtype=np.float32) * .1)
    obj._check_audio_health(now=20)
    assert 'arriving again' in capsys.readouterr().out


def test_callback_exception_is_not_silenced(capsys):
    obj = listener()
    class BrokenInput:
        def copy(self):
            raise RuntimeError('capture buffer failed')
    obj._on_audio(BrokenInput(), 960, None, None)
    obj._check_audio_health(now=6)
    assert 'capture buffer failed' in capsys.readouterr().out


def _capture_listener():
    import threading
    obj = listener(16000)
    obj._utterance_q = queue.Queue(maxsize=8)
    obj._utterances_dropped = 0
    obj._vad_seconds = 0.004
    obj._processing_thread = None
    obj._check_query_timeout = lambda: None
    obj.echo_detector = SimpleNamespace(_utterance_start_time=1.0, track_utterance_timing=lambda *a: None)
    obj._frame_ms = 20
    obj._pre_roll = deque()
    obj.tts = None
    obj.is_speech_active = True
    obj._silence_frames = 5
    obj._utterance_frames = [np.ones(obj._frame_samples, dtype=np.float32) * .1] * 10
    return obj, threading


def test_capture_does_not_wait_for_speech_recognition():
    """Cutting an utterance returns at once even while Whisper is busy."""
    import time
    obj, threading = _capture_listener()
    release = threading.Event()
    transcribed = []

    def slow_transcribe(utt):
        release.wait(5)
        transcribed.append(utt)

    obj._transcribe_utterance = slow_transcribe
    obj._processing_thread = threading.Thread(target=obj._processing_worker_loop, daemon=True)
    obj._processing_thread.start()

    started = time.monotonic()
    obj._finalize_utterance()
    assert time.monotonic() - started < 0.1
    assert obj._utterance_frames == [] and not obj.is_speech_active

    release.set()
    obj._utterance_q.put(None)
    obj._processing_thread.join(5)
    assert len(transcribed) == 1
    assert transcribed[0].audio.size == obj._frame_samples * 10
    assert transcribed[0].samplerate == 16000
    assert transcribed[0].start_time == 1.0


def test_recognition_backlog_drops_oldest_utterance_visibly(capsys):
    obj, threading = _capture_listener()
    release = threading.Event()
    seen = []

    def blocked_transcribe(utt):
        release.wait(5)
        seen.append(utt.start_time)

    obj._transcribe_utterance = blocked_transcribe
    obj._processing_thread = threading.Thread(target=obj._processing_worker_loop, daemon=True)
    obj._processing_thread.start()

    from jarvis.listening.listener import _Utterance
    for i in range(12):
        obj._submit_utterance(_Utterance(np.zeros(10), 16000, float(i), float(i), 0.0, False))

    release.set()
    obj._utterance_q.put(None)
    obj._processing_thread.join(5)
    assert seen[-1] == 11.0
    assert len(seen) < 12
    assert 'falling behind' in capsys.readouterr().out


def test_without_a_worker_utterances_are_transcribed_inline():
    obj, _ = _capture_listener()
    transcribed = []
    obj._transcribe_utterance = transcribed.append
    obj._finalize_utterance()
    assert len(transcribed) == 1


def test_length_limit_cut_repeats_the_last_second_in_the_next_utterance():
    """A forced cut right after the wake word must not strand it."""
    obj, _ = _capture_listener()
    frames = [np.full(obj._frame_samples, i, dtype=np.float32) for i in range(100)]
    obj._utterance_frames = list(frames)
    transcribed = []
    obj._transcribe_utterance = transcribed.append

    obj._end_utterance(forced=True)

    assert transcribed[0].audio.size == obj._frame_samples * 100
    assert obj.is_speech_active
    assert len(obj._utterance_frames) == 50  # 1 s of 20 ms frames
    assert obj._utterance_frames[0][0] == 50


def test_endpoint_cut_starts_the_next_utterance_empty():
    obj, _ = _capture_listener()
    obj._transcribe_utterance = lambda utt: None

    obj._end_utterance(forced=False)

    assert obj._utterance_frames == [] and not obj.is_speech_active


def test_length_limit_cut_during_playback_does_not_overlap():
    obj, _ = _capture_listener()
    obj.tts = SimpleNamespace(is_speaking=lambda: True)
    obj._transcribe_utterance = lambda utt: None

    obj._end_utterance(forced=True)

    assert obj._utterance_frames == [] and not obj.is_speech_active


def _frame_listener(tts_speaking=False):
    obj, _ = _capture_listener()
    obj.is_speech_active = False
    obj._utterance_frames = []
    obj._silence_frames = 0
    obj._speech_frames_seen = 0
    obj._pre_roll_max_frames = 12
    obj._endpoint_silence_frames = 40
    obj._normal_max_utt_frames = 600
    obj._tts_max_utt_frames = 150
    obj.tts = SimpleNamespace(is_speaking=lambda: tts_speaking)
    obj._is_speech_frame = lambda frame: True
    cut = []
    obj._transcribe_utterance = cut.append
    return obj, cut


def test_continuous_speech_during_playback_is_cut_every_three_seconds():
    """Jarvis's own voice leaves no pauses; a stop command must not wait for one."""
    obj, cut = _frame_listener(tts_speaking=True)
    frame = np.ones(obj._frame_samples, dtype=np.float32) * .1
    for _ in range(450):  # 9 s of unbroken speech
        obj._process_capture_frame(frame)
    assert len(cut) == 3
    assert all(u.audio.size == obj._frame_samples * 150 for u in cut)


def test_unbroken_room_noise_is_cut_at_the_length_limit():
    obj, cut = _frame_listener()
    frame = np.ones(obj._frame_samples, dtype=np.float32) * .1
    for _ in range(601):
        obj._process_capture_frame(frame)
    assert len(cut) == 1
    assert len(obj._utterance_frames) == 51  # 1 s overlap carried + the next frame


def test_speech_ends_at_endpoint_silence():
    obj, cut = _frame_listener()
    frame = np.ones(obj._frame_samples, dtype=np.float32) * .1
    for _ in range(50):
        obj._process_capture_frame(frame)
    obj._is_speech_frame = lambda frame: False
    for _ in range(40):
        obj._process_capture_frame(frame)
    assert len(cut) == 1
    assert not obj.is_speech_active and obj._utterance_frames == []
