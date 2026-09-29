"""Ollama implementation of :class:`LLMBackend`.

HTTP calls against ``/api/chat``, ``/api/embed`` and ``/api/tags``,
native tool calling with the ``tools`` parameter (Ollama 0.4+), and
fail-soft error handling (return ``None`` on timeouts / connection
errors; :class:`ToolsNotSupportedError` on HTTP 400 with tools).

Every request carries the same ``num_ctx`` and the backend's
``keep_alive``. Ollama reloads a model runner whenever a request asks
for a different context size, and a request without ``keep_alive``
resets residency to the server default of five minutes, so letting
call sites vary either knob causes multi-second cold reloads and
discards the prompt KV cache between turns.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import json
import requests

from ..debug import debug_log
from .errors import is_timeout_error
from .backend import LLMBackend, ToolsNotSupportedError, strip_nonstandard_message_fields


def check_version(base_url: str, timeout: float = 5.0) -> tuple[bool, str | None]:
    """Probe ``GET /api/version`` and return ``(True, version_str)`` if the
    endpoint responds as an Ollama server, or ``(False, None)`` on failure or
    non-Ollama response."""
    try:
        resp = requests.get(f"{base_url}/api/version", timeout=timeout)
        if resp.status_code != 200:
            return False, None
        data = resp.json()
        if not isinstance(data, dict):
            return False, None
        version = data.get("version")
        if not isinstance(version, str) or not version:
            return False, None
        return True, version
    except Exception:
        return False, None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from an LLM chat response across known shapes.

    Handles Ollama's ``message.content`` shape and the OpenAI-style
    ``choices[0].message.content`` / ``choices[0].text`` fallbacks so
    callers do not need to special-case responses that come back from
    OpenAI-compatible runtimes proxied through Ollama.
    """
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content

    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content

    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content

    return None


OLLAMA_NUM_CTX = 8192
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
_BACKEND_OWNED_KEYS = frozenset({"num_ctx", "keep_alive"})


class OllamaBackend(LLMBackend):
    """:class:`LLMBackend` implementation that talks to a local Ollama server."""

    def __init__(self, base_url: str, keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE) -> None:
        self._base_url = base_url.rstrip("/")
        self._keep_alive = keep_alive

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def keep_alive(self) -> str:
        return self._keep_alive

    def _payload(self, model: str, messages: List[Dict[str, Any]], *, stream: bool,
                 thinking: bool) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "cache_prompt": True,
            "keep_alive": self._keep_alive,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
            "think": thinking,
        }

    # ── chat ───────────────────────────────────────────────────────────

    def direct(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        timeout_sec: float = 10.0,
        thinking: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Direct LLM call without temporal context, location, or other
        ``ask_coach`` features.

        ``temperature`` is forwarded to Ollama when set. Pass ``0.0``
        for classification / extraction calls where determinism beats
        creativity — Ollama defaults to ~0.8 otherwise, which can
        flake small models on rule-following tasks (e.g. the knowledge
        extractor's banned-form list).

        ``max_tokens`` maps to Ollama's ``num_predict``, capping the
        total generated tokens (including reasoning). Essential for
        classification calls where small reasoning models otherwise
        loop endlessly.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload = self._payload(chat_model, messages, stream=False, thinking=thinking)
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            with requests.post(
                f"{self._base_url}/api/chat", json=payload, timeout=timeout_sec
            ) as resp:
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, dict):
                content = extract_text_from_response(data)
                if isinstance(content, str) and content.strip():
                    return content
                debug_log(
                    f"OllamaBackend.direct: empty content from response keys={list(data.keys())}",
                    "llm",
                )
        except requests.exceptions.Timeout:
            debug_log(f"OllamaBackend.direct: timeout after {timeout_sec}s", "llm")
            return None
        except Exception as e:
            debug_log(f"OllamaBackend.direct: request failed — {e}", "llm")
            return None

        return None

    def streaming(
        self,
        chat_model: str,
        system_prompt: str,
        user_content: str,
        on_token: Optional[Callable[[str], None]] = None,
        timeout_sec: float = 30.0,
        thinking: bool = False,
    ) -> Optional[str]:
        """Streaming LLM call that invokes ``on_token`` for each token
        received. Returns the complete response text, or ``None`` on
        error / empty stream.

        Uses ``with requests.post(...)`` so the streaming response (and
        the underlying TCP connection) is released even if iteration
        exits early via an exception or the caller stops consuming.
        Without this, an aborted stream pinned the connection until GC,
        which could happen many turns later under sustained reply
        load.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload = self._payload(chat_model, messages, stream=True, thinking=thinking)

        try:
            with requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout_sec,
                stream=True,
            ) as resp:
                resp.raise_for_status()

                full_response: List[str] = []
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and isinstance(data["message"], dict):
                                content = data["message"].get("content", "")
                                if content:
                                    full_response.append(content)
                                    if on_token:
                                        on_token(content)
                        except json.JSONDecodeError:
                            continue

                result = "".join(full_response)
                return result if result.strip() else None

        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None

    def chat(
        self,
        chat_model: str,
        messages: List[Dict[str, Any]],
        timeout_sec: float = 30.0,
        extra_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Send an arbitrary messages array to Ollama and return the
        raw response JSON. Caller is responsible for interpreting
        assistant content (including JSON / tool calls).

        The pinned ``num_ctx`` (8192) keeps the system prompt (tool list +
        protocol guidance + memory context) from overflowing and forcing
        Ollama to truncate the tool schema — small models like
        ``gemma4:e2b`` then fall back to pre-trained ``tool_code``
        scaffolding instead of producing valid tool calls.
        """
        sanitised = strip_nonstandard_message_fields(messages)
        payload = self._payload(chat_model, sanitised, stream=False, thinking=thinking)
        # ``extra_options`` keys land at the Ollama wire root for known
        # request-level fields (``format``, ``think``); the rest fold into
        # the sampling-options dict. ``max_tokens`` is the canonical
        # generation cap across backends — translate it to Ollama's
        # ``num_predict`` so callers don't need to know which knob each
        # server speaks. ``num_ctx`` and ``keep_alive`` are owned by the
        # backend and ignored here.
        if extra_options and isinstance(extra_options, dict):
            for key, value in extra_options.items():
                if key in _BACKEND_OWNED_KEYS:
                    continue
                if key in {"format", "think"}:
                    payload[key] = value
                elif key == "max_tokens":
                    payload["options"]["num_predict"] = int(value)
                elif key == "options" and isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        if inner_key in _BACKEND_OWNED_KEYS:
                            continue
                        if inner_key == "max_tokens":
                            payload["options"]["num_predict"] = int(inner_value)
                        else:
                            payload["options"][inner_key] = inner_value
                else:
                    payload["options"][key] = value

        if tools and isinstance(tools, list) and len(tools) > 0:
            payload["tools"] = tools

        try:
            with requests.post(
                f"{self._base_url}/api/chat", json=payload, timeout=timeout_sec
            ) as resp:
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                return data
        except requests.exceptions.Timeout:
            print(f"  ⏱️ LLM request timed out (configured timeout: {timeout_sec:g}s)", flush=True)
            return None
        except requests.exceptions.ConnectionError as exc:
            if is_timeout_error(exc):
                debug_log("chat response read timed out (wrapped transport timeout)", "llm")
                print(f"  ⏱️ LLM request timed out (configured timeout: {timeout_sec:g}s)", flush=True)
                return None
            # Bubble out so callers (e.g. the intent judge) can distinguish
            # "server unreachable" from a transient error and apply their own
            # back-off policy.
            print("  ❌ LLM connection error", flush=True)
            raise
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and tools:
                raise ToolsNotSupportedError(
                    f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
                )
            status = e.response.status_code if e.response is not None else "?"
            print(f"  ❌ LLM HTTP error (status {status})", flush=True)
            return None
        except Exception as e:
            print(f"  ❌ LLM error ({type(e).__name__})", flush=True)
            return None

        return None

    # ── embeddings & discovery ────────────────────────────────────────

    def embed(
        self,
        text: str,
        model: str,
        timeout_sec: float = 15.0,
    ) -> Optional[List[float]]:
        """Embed ``text`` via Ollama's ``/api/embed``.

        Servers older than 0.3.4 lack ``/api/embed`` and answer 404; those
        fall back to the legacy ``/api/embeddings`` endpoint. Both vectors
        are usable interchangeably because the vector store L2-normalises
        every vector before indexing and searching.
        """
        payload = {"model": model, "input": text, "keep_alive": self._keep_alive}
        try:
            resp = requests.post(
                f"{self._base_url}/api/embed", json=payload, timeout=timeout_sec,
            )
            if resp.status_code == 404:
                resp = requests.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": model, "prompt": text, "keep_alive": self._keep_alive},
                    timeout=timeout_sec,
                )
                resp.raise_for_status()
                vec = resp.json().get("embedding")
            else:
                resp.raise_for_status()
                vectors = resp.json().get("embeddings")
                vec = vectors[0] if isinstance(vectors, list) and vectors else None
            if isinstance(vec, list) and vec:
                return [float(x) for x in vec]
            debug_log(f"OllamaBackend.embed: no vector in response (model={model})", "llm")
        except requests.exceptions.Timeout:
            debug_log(f"OllamaBackend.embed: timeout after {timeout_sec}s (model={model})", "llm")
        except Exception as e:
            debug_log(f"OllamaBackend.embed: {type(e).__name__}: {e}", "llm")
        return None

    def list_models(self, timeout_sec: float = 5.0) -> List[str]:
        """List installed Ollama models via ``GET /api/tags``."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", []) if isinstance(data, dict) else []
            names: List[str] = []
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name")
                    if isinstance(name, str) and name:
                        names.append(name)
            return names
        except Exception:
            return []

    def warm_up(
        self,
        model: str,
        timeout_sec: float = 60.0,
    ) -> bool:
        """Probe ``/api/version`` to verify the server is Ollama, then issue a
        minimal ``/api/chat`` request so it loads ``model`` into resident memory
        for the backend's ``keep_alive`` duration. The chat-endpoint warmup
        exercises the full inference pipeline (JIT compilation, KV-cache
        allocation) that an empty ``/api/generate`` would not trigger,
        preventing a timeout on the first real intent-judge or reply-engine
        call. It uses the same pinned ``num_ctx`` as real requests so the
        runner it loads is the one they reuse. Best-effort: errors are
        swallowed so callers never crash on warmup failure."""
        if not self._base_url or not model:
            return False
        try:
            # Verify the server is actually Ollama before warming up —
            # a non-Ollama HTTP server on the same port could return 200
            # to a chat POST and produce a false positive.
            version_to = min(timeout_sec, 5.0)
            ok, _ = check_version(self._base_url, timeout=version_to)
            if not ok:
                return False

            remaining = max(1.0, timeout_sec - version_to)
            payload = self._payload(
                model,
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "ping"},
                ],
                stream=False,
                thinking=False,
            )
            payload["options"].update({"num_predict": 1, "temperature": 0.0})
            resp = requests.post(
                f"{self._base_url}/api/chat", json=payload, timeout=remaining,
            )
            return resp.status_code == 200
        except Exception:
            return False
