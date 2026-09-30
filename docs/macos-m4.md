# Jarvis on an M4 MacBook Air (16 GB)

This page records the configuration this fork is developed and measured on, what was verified on it, and what is still known to fall short. Every figure comes from a real run on the machine below; nothing here is a projection.

Paths use `~/AI/jarvis` as an example checkout location. Use whatever directory you prefer.

## Hardware and software

| Component | Value |
| :--- | :--- |
| Machine | MacBook Air, Apple M4 |
| Memory | 16 GB unified memory |
| Architecture | arm64 |
| OS | macOS 27.0 |
| Python | 3.12, in a virtual environment at `.venv` inside the checkout |
| Model server | Ollama, local, bound to `127.0.0.1:11434` only |

## Models

| Role | Model | Notes |
| :--- | :--- | :--- |
| Chat | `gemma4:e2b` | Detected as a small model, so tools are described in the prompt as text rather than through the native tools API. |
| Fast model (voice intent, tool routing) | `gemma4:e2b` | Shares the chat model's loaded runner, so no second model has to stay resident. |
| Embeddings | `nomic-embed-text` | 768 dimensions; 11 to 37 ms per request. |
| Speech recognition | MLX Whisper `base` | Runs on the Apple Silicon GPU when run from source; 141 to 228 ms per utterance. |
| Speech synthesis | Piper | Streams playback sentence by sentence. |

`qwen3.5:0.8b` was tested as the fast model and not adopted. On the voice-intent evaluation it scored 42/47 against 46/47 for `gemma4:e2b` (it wrongly treated ambient speech and past-tense narrative as directed at Jarvis), took longer overall (175 s against 152 s), and would need about 1.2 GB of extra resident memory on a machine that is already using swap.

Model files are downloaded by Ollama and Whisper at runtime and are never part of this repository.

## Configuration

Settings live in `~/.config/jarvis/config.json`. The values that differ from the defaults on this machine:

```json
{
  "ollama_chat_model": "gemma4:e2b",
  "fast_model": "gemma4:e2b",
  "whisper_model": "base",
  "voice_collect_seconds": 1.5,
  "hot_window_seconds": 6.0
}
```

- `whisper_model: base` trades some accuracy for speed and memory; the default is `medium`.
- `voice_collect_seconds: 1.5` stops listening 1.5 s after you finish speaking. The default of 4.5 s adds about 3 s to every reply.
- `hot_window_seconds: 6.0` gives you 6 s after a reply to ask a follow-up without saying "Jarvis".

### Ollama server settings

These are set in the Ollama launch agent's environment, not in Jarvis:

| Variable | Value | Why |
| :--- | :--- | :--- |
| `OLLAMA_FLASH_ATTENTION` | `1` | Faster attention on Apple Silicon. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Halves KV-cache memory. |
| `OLLAMA_NUM_PARALLEL` | `4` | Lets the intent judge, router and chat share one loaded model instead of queueing. |

Jarvis pins every Ollama request to an 8192-token context. A request with a different context size makes Ollama load a second copy of the model, which on 16 GB stalls the machine.

Keep Ollama bound to `127.0.0.1`. Do not expose it on your network.

## What was verified

Results from live voice sessions on this machine, speaking to the running daemon:

| Capability | Result |
| :--- | :--- |
| Wake word | 8 of 8 queries accepted. |
| Follow-up without the wake word | 2 of 2 answered correctly inside the follow-up window. |
| Audio capture | No dropped audio blocks across about 11 minutes of capture. |
| Memory across restarts | A fact stored by voice was recalled correctly after a full daemon restart. |
| Chrome automation (direct tool call) | 30 tools discovered; opening a page worked. |
| macOS automation (direct tool call) | A read-only AppleScript ran successfully. |
| Unit tests | Full suite green on the published code (see the README's Testing section). |

## Latency

Measured from the end of speech to the first audio, before streamed speech was added:

| Stage | Time |
| :--- | :--- |
| Speech recognition | 0.14 to 0.23 s |
| Voice intent decision | 0.9 to 1.6 s |
| Collection window | 1.6 to 1.7 s (with `voice_collect_seconds: 1.5`) |
| Reply generation | 3.4 to 20.6 s |
| Total | 7.0 to 24.3 s |

Reply generation dominates, and it is longest when the planner decides to search the web. Replies now start playing as soon as the first sentence is ready, which shortens the wait to first audio; that change has unit tests but has not yet been re-measured live.

## Known limitations on this machine

- **Barge-in over the laptop speakers does not work.** Saying "stop" while Jarvis is speaking was not detected in 0 of 4 live attempts: without acoustic echo cancellation, Whisper hears Jarvis's own voice louder than yours. The stop mechanics pass their unit tests. A headset avoids the problem.
- **Tool arguments from the small model are sometimes wrong.** For example, asking Chrome to navigate to a site produced an argument the Chrome tool rejected.
- **Memory pressure.** With `gemma4:e2b` and `nomic-embed-text` loaded, swap use reached 3.7 GB of 4 GB.
- **The packaged app uses CPU speech recognition.** The current app bundle ships faster-whisper rather than MLX Whisper, so run from source for GPU speech recognition.
- **Dictation is unavailable on macOS 26 and later** because of a pynput incompatibility.
- **Location needs a GeoLite2 database.** Without it, weather requests need a city.
- **Keep the checkout out of iCloud-synced folders** such as Desktop or Documents when iCloud Drive syncs them. File syncing of the checkout and `.venv` made the test suite run about four times slower here.
