# Jarvis

**A personal, local-first Jarvis AI assistant optimised for Apple Silicon Macs.**

Voice-first, private by default, and free to run: speech recognition, language models and speech synthesis all run on your own Mac.

<p align="center">
  <img src="docs/img/face.png" alt="Jarvis's animated amber wireframe face, the desktop presence of the voice assistant" width="460">
</p>

> This repository is a customised fork of [isair/jarvis](https://github.com/isair/jarvis) by Baris Sencan. Most of the application is upstream's work; this fork adds tuning and fixes for M4-class MacBooks. See [Upstream](#upstream) and [License](#license).
>
> **Status: development checkpoint (`m4-checkpoint-1`), not a stable release.** It is usable day to day on the machine it was tuned on, but it is not production-ready. See [Project status and known limitations](#project-status-and-known-limitations).

## Overview

Jarvis listens for its name, understands what you ask, and answers out loud. It remembers what you tell it across sessions, can search the web and check the weather, and can drive Chrome and macOS through MCP tools. Everything except web lookups runs locally through [Ollama](https://ollama.com), MLX Whisper and Piper.

This fork focuses on making that experience fast and dependable on a 16 GB MacBook Air M4: lower voice latency, streamed speech, audio capture that never blocks, and a model setup that fits in memory. The validated configuration is documented in [docs/macos-m4.md](docs/macos-m4.md).

Status markers used throughout this README:

- ✅ **Working**: verified on the M4 machine described in [docs/macos-m4.md](docs/macos-m4.md).
- ⚠️ **Experimental**: implemented and unit-tested, but with known gaps or not yet verified live.
- 🚧 **Planned**: not implemented yet.

## Features

| Feature | Status |
| :--- | :--- |
| Wake word ("Jarvis" anywhere in a sentence) | ✅ 8 of 8 live queries accepted |
| Follow-up questions without repeating the wake word | ✅ within a configurable follow-up window |
| Local chat with Ollama (`gemma4:e2b`) | ✅ |
| Speech recognition with MLX Whisper on the Apple Silicon GPU | ✅ when run from source |
| Speech synthesis with Piper, played sentence by sentence | ✅ |
| Speaking a reply while it is still being generated | ⚠️ unit-tested, not yet re-measured live |
| Persistent memory (diary and knowledge graph) | ✅ recalled correctly after a restart |
| Web search and weather | ⚠️ weather needs a city unless a GeoLite2 database is installed |
| Multi-turn tool use (follow-ups and topic changes) | ⚠️ known failures with the small model, see [Testing](#testing) |
| Chrome automation through MCP | ⚠️ direct tool calls work; voice commands sometimes produce arguments the tool rejects |
| macOS automation through MCP | ⚠️ direct tool calls work; not yet tested by voice |
| Interrupting Jarvis by saying "stop" (barge-in) | ⚠️ works with a headset; fails over laptop speakers |
| Desktop app (tray, animated face, chat window, settings) | ✅ inherited from upstream |
| Packaged macOS app with this fork's changes | ⚠️ builds with upstream's scripts; bundle still uses CPU speech recognition |
| Global dictation hotkey | ⚠️ unavailable on macOS 26 and later (upstream issue) |
| Confirmation before destructive actions | 🚧 planned |
| Acoustic echo cancellation | 🚧 planned |
| `jarvis doctor` health check | 🚧 planned |

## Architecture

```
microphone ─► capture thread ─► voice activity detection ─► Whisper ─► intent judge
                                                                         │
                     ┌───────────────────────────────────────────────────┘
                     ▼
               reply engine: tool router ─► planner ─► memory lookup ─► chat model ─► tools
                     │
                     ▼
               speech stream ─► Piper ─► speakers
```

- **Listener** (`src/jarvis/listening/`): captures audio on its own thread, detects speech, transcribes it, and asks a small model whether the speech was directed at Jarvis.
- **Reply engine** (`src/jarvis/reply/`): routes the request to a small set of relevant tools, plans the steps, pulls relevant memories, runs the chat model and executes tool calls.
- **Memory** (`src/jarvis/memory/`): a SQLite diary of past conversations and a knowledge graph of facts about you.
- **Tools** (`src/jarvis/tools/`): built-in tools plus any MCP servers you configure.
- **Output** (`src/jarvis/output/`): Piper or Chatterbox speech synthesis.
- **Desktop app** (`src/desktop_app/`): PyQt6 tray app, face, chat, settings and memory viewer.

<p align="center">
  <img src="docs/img/chat-window.png" alt="Jarvis companion chat window with illustrative messages and an amber composer" width="480">
</p>

Each module has a `*.spec.md` file describing its behaviour, and [docs/llm_contexts.md](docs/llm_contexts.md) maps every model call.

## Voice Pipeline

1. Audio is captured continuously on a dedicated thread; transcription and replies run on worker threads, so capture never stalls.
2. After you stop speaking, Jarvis waits for a short silence (`voice_collect_seconds`), then transcribes with Whisper.
3. The intent judge decides whether the speech was meant for Jarvis, using a rolling transcript of recent speech for context.
4. The reply streams from the chat model; each complete sentence goes to Piper as soon as it is ready.
5. After replying, Jarvis keeps listening for a follow-up for `hot_window_seconds` without needing the wake word.
6. Saying "stop" while Jarvis speaks cancels both playback and generation (reliable only with a headset for now).

Each voice turn logs a timing line such as `VOICE capture=0ms vad=18ms whisper=205ms intent=1033ms collect=1625ms chat=... total=...`, and each reply logs its stage timings, so latency regressions are visible.

## Local AI

All inference runs on your Mac through Ollama, bound to `127.0.0.1`. An OpenAI-compatible local server (LM Studio, llama.cpp and others) also works. No cloud AI account is needed.

## Models

| Role | Model on the M4 | Notes |
| :--- | :--- | :--- |
| Chat | `gemma4:e2b` | Upstream default; fits comfortably in 16 GB. |
| Fast model (intent, routing) | `gemma4:e2b` | Shares the chat model's memory. `qwen3.5:0.8b` was tested and scored lower. |
| Embeddings | `nomic-embed-text` | Semantic memory search. |
| Speech recognition | MLX Whisper `base` | `medium` is the default and more accurate but slower. |
| Speech synthesis | Piper | Downloaded on first use. |

Only model names are stored in this repository. Model weights are downloaded by Ollama and Whisper on first use and are never committed.

## Memory

Jarvis keeps two kinds of memory on your Mac under `~/.local/share/jarvis`:

- **Diary**: summaries of past conversations, searchable by keyword and, with embeddings, by meaning.
- **Knowledge graph**: facts about you and the world, grouped into branches. Facts about you are included in every reply.

Sensitive information such as email addresses and API keys is redacted before it reaches the model or the diary. The desktop app's Memory Viewer shows everything stored. Memory databases are never part of this repository.

## MCP / Tools

Built-in tools: web search (DuckDuckGo), web page fetch, weather (Open-Meteo), time, screenshot OCR, local files (home folder only), nutrition logging and tool search. MCP servers add more; a small router picks the relevant tools for each request so the small model is not overwhelmed. Configure MCP servers in `~/.config/jarvis/config.json` under `mcps`, as described in [docs/CONFIGURATION.md](docs/CONFIGURATION.md#mcp-integrations).

## Chrome Automation

⚠️ Uses [chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp), which launches its own Chrome instance and does not touch your normal browser profile:

```json
"mcps": {
  "chrome-devtools": { "transport": "stdio", "command": "npx", "args": ["-y", "chrome-devtools-mcp@latest"] }
}
```

Verified: 30 tools discovered and pages opened through direct tool calls. By voice, the small model sometimes passes the wrong argument (for example a page name instead of a URL).

## macOS Automation

⚠️ Uses [macos-automator-mcp](https://github.com/steipete/macos-automator-mcp) to run AppleScript and JXA:

```json
"mcps": {
  "macos": { "transport": "stdio", "command": "npx", "args": ["-y", "@steipete/macos-automator-mcp"] }
}
```

Verified with a direct, read-only script. This tool can run any script the model produces and there is no confirmation step yet, so enable it only if you accept that risk.

## Privacy

- Voice, transcripts, memory and model inference stay on your Mac.
- Network access happens for web search, web page fetches, weather, location detection, model downloads, update checks and any MCP servers you enable.
- Sensitive values are redacted before they reach model context or the diary.
- To reduce network access further, see the privacy options in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Cost

₹0 per month to run. All models are free and local, and web search uses DuckDuckGo without an API key. The only costs are electricity and disk space for the models (about 3 GB for the configuration above).

## Requirements

- Apple Silicon Mac (developed on an M4 with 16 GB; 8 GB machines need smaller models)
- macOS (developed and tested on macOS 27.0; older versions are untested)
- Python 3.12
- [Ollama](https://ollama.com/download)
- Node.js, only for the MCP servers above (they run through `npx`)
- A microphone; a headset is recommended so Jarvis does not hear itself

## macOS Installation

Upstream publishes signed app builds on its [releases page](https://github.com/isair/jarvis/releases). Those builds do not include this fork's changes. To use this fork, run it from source as described below. No binaries are published from this repository yet.

## Development Setup

```bash
git clone https://github.com/Deepanshxsharma/jarvis.git ~/AI/jarvis
cd ~/AI/jarvis
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Keep the checkout out of iCloud-synced folders (such as Desktop or Documents with iCloud Drive enabled); syncing the checkout and `.venv` slows everything down considerably.

## Running from Source

```bash
cd ~/AI/jarvis
source .venv/bin/activate
PYTHONPATH=src python -m jarvis.daemon        # voice assistant in the terminal
bash scripts/run_desktop_app.sh               # desktop app with tray, face and chat
```

`bash scripts/run_macos.sh` creates the environment, installs dependencies and starts the daemon in one step.

## Ollama Setup

```bash
brew install ollama
brew services start ollama
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

Optional server settings that help on 16 GB machines are listed in [docs/macos-m4.md](docs/macos-m4.md#ollama-server-settings). Keep Ollama listening on `127.0.0.1` only.

## Voice Setup

1. Grant microphone access to your terminal (or to the app) when macOS asks.
2. MLX Whisper and the Piper voice download automatically on first run; watch the log for progress.
3. Set `"whisper_model": "base"` for speed or leave the default `medium` for accuracy.
4. Say "Jarvis, what time is it?" once the log reports that Jarvis is listening.

## Configuration

Settings live in `~/.config/jarvis/config.json` and can be edited through the desktop app's Settings window. [examples/config.json](examples/config.json) lists every option, and [docs/CONFIGURATION.md](docs/CONFIGURATION.md) explains them. The M4 values are in [docs/macos-m4.md](docs/macos-m4.md#configuration).

## Troubleshooting

- **Replies take a long time**: most of the time is the chat model, especially when it searches the web. Check the `⏱️ REPLY` timing line to see which stage is slow.
- **Jarvis does not respond to "stop" while speaking**: use a headset; without echo cancellation, Whisper hears Jarvis's own voice.
- **Weather asks for a city**: install a GeoLite2 database with `python scripts/setup_geolocation.py`, or name the city.
- **The Mac swaps heavily**: close other large apps, or use a smaller Whisper model.
- **Downloads look stuck**: first-run model downloads can take several minutes; check the log.

More in [docs/CONFIGURATION.md](docs/CONFIGURATION.md#troubleshooting).

## Performance

Measured on the M4 before streamed speech was added: speech recognition takes 0.14 to 0.23 s, the intent decision 0.9 to 1.6 s, and reply generation 3.4 to 20.6 s, for 7 to 24 s from the end of speech to the first audio. Full figures and the conditions are in [docs/macos-m4.md](docs/macos-m4.md#latency).

Changes in this fork that target latency:

- Ollama requests always use the same context size and residency, so models are not reloaded mid-conversation.
- Transcription and replies run off the audio capture thread.
- Piper plays each sentence as soon as it is synthesised, and replies are spoken while they are generated.
- Each reply reports how long every stage took.

## Security

- Jarvis opens no listening network ports; Ollama listens on `127.0.0.1` only.
- The local files tool is restricted to your home folder, but it can overwrite and delete files there without asking.
- MCP tools run with your user's permissions. The macOS automation server can run arbitrary AppleScript.
- 🚧 There is no confirmation step before destructive actions yet. Only enable tools you are comfortable with the model using unsupervised.
- Secrets in your config (API keys for optional services) stay in `~/.config/jarvis/config.json`, outside this repository.

## Testing

```bash
source .venv/bin/activate
python -m pytest -q tests                  # unit tests, no models needed
python -m pytest evals -m eval             # behaviour evaluations against a running Ollama
```

The unit suite passes on this fork (2569 tests at `m4-checkpoint-1`). The live evaluations run the real model and are not all green: the most recent full run had 52 passed, 4 failed and 5 expected failures with `gemma4:e2b`. The four failures are described in [Project status and known limitations](#project-status-and-known-limitations). See [EVALS.md](EVALS.md) for the evaluation design.

## Project status and known limitations

`m4-checkpoint-1` is a development checkpoint: the M4 tuning, memory grounding and routing work validated so far, tagged so later changes can be compared against it. It is not a stable release, no binaries are published for it, and the upstream release workflows are disabled in this repository.

Known limitations at this checkpoint:

- **Unknown film lookup ("Tell me about the movie Possessor")**: the planner, which runs at temperature 0, plans a `fetchWebPage` step with a search phrase instead of a URL. The fetch fails and the reply is answered from the model's own knowledge instead of a web search. The matching evaluation fails every time.
- **Store opening hours after a weather question**: asking how long a shop is open straight after a weather reply is still routed to the weather tool instead of a web search.
- **"Say something" grounding evaluation**: it depends on a fake memory store in the evaluation harness that cannot serve stored facts, so it fails there even though grounding in stored facts works against a real memory database.
- **Live weather evaluation**: it depends on the MaxMind GeoLite2 fixture installed on the test machine, so its result varies with that fixture. The small model can also state a temperature it never looked up.
- **Qt tests on an iCloud-synced folder**: when the repository lives on an iCloud-synced Desktop, macOS marks files in `.venv` as hidden, Qt stops finding its platform plugins, and Qt tests fail at random. Keep the checkout outside iCloud-synced folders.
- **Residual nondeterminism**: routing and planning calls run at temperature 0, but Ollama's prompt cache can still flip close decisions, and the chat model samples at its default temperature, so some live evaluations pass on some runs and not others.

## Building the macOS App

```bash
source .venv/bin/activate
bash scripts/build_installer.sh            # produces dist/Jarvis.app
```

⚠️ The bundle currently ships CPU speech recognition (faster-whisper) rather than MLX Whisper, so it is slower than running from source. Built apps are not committed to this repository; binaries belong on a releases page.

## Upstream

This fork tracks [isair/jarvis](https://github.com/isair/jarvis), created by Baris Sencan. It is a customised fork focused on:

- Apple Silicon and M4 performance
- local-first operation
- voice latency
- personal memory
- tool routing
- computer automation

To pull in upstream changes:

```bash
git fetch upstream
git merge upstream/main
```

Please report issues that also affect upstream to [isair/jarvis](https://github.com/isair/jarvis/issues).

## License

Jarvis is distributed under the [Jarvis AI Assistant License](LICENSE), copyright (c) 2025 Baris Sencan. It permits personal, educational and other **non-commercial** use, modification and distribution, provided the copyright and permission notice are kept and derivative works use the same terms. Commercial use needs a separate licence from the copyright holder. This fork is published under the same licence.
