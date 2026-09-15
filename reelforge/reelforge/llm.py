"""Pluggable writing models.

Same pattern as the speech backends: a name maps to a function, and `auto` picks
the best one actually available. Three providers ship:

* `claude`  - the Anthropic API. Much the best Arabic, costs a few cents a script.
* `ollama`  - a model on your own machine. Free and fully private, weaker Arabic.
* `stub`    - no model at all. Assembles a script skeleton from the framework and
              the fact sheet so the pipeline works, and the tests run, with no key
              and no network.

Nothing here decides *what* is true. Numbers come from `market.py` and are handed
to the model as fixed text; `script.py` then audits the output against them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CLAUDE_MODEL = "claude-opus-5"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("REELFORGE_OLLAMA_MODEL", "qwen2.5:14b")


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    data: dict | None = None          # parsed JSON when a schema was requested
    usage: dict = field(default_factory=dict)


# ------------------------------------------------------------------- claude

def _anthropic_credentials_present() -> bool:
    """Env var, auth token, or a profile written by `ant auth login`."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "anthropic"
    return config.exists() and any(config.iterdir())


def generate_claude(prompt: str, *, system: str, schema: dict | None = None,
                    model: str | None = None, max_tokens: int = 16000) -> LLMResponse:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise LLMUnavailable(
            "the anthropic package is not installed. Run:\n"
            "  pip install anthropic\n"
            "or write locally instead with --provider ollama."
        ) from exc

    model = model or CLAUDE_MODEL
    client = anthropic.Anthropic()

    request: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "adaptive"},
    }
    if schema:
        request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

    # Server-side fallback keeps a declined request working by re-running it on
    # another model. Older SDKs do not know the parameter, so fall back to a plain
    # call rather than failing outright.
    try:
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
        )
    except (TypeError, AttributeError):
        response = client.messages.create(**request)
    except Exception as exc:                      # unknown beta/param on this account
        if type(exc).__name__ not in ("BadRequestError", "NotFoundError"):
            raise
        response = client.messages.create(**request)

    if getattr(response, "stop_reason", None) == "refusal":
        raise LLMError("the model declined this request. Try rephrasing the topic.")

    text = "".join(block.text for block in response.content
                   if getattr(block, "type", None) == "text")
    usage = {}
    if getattr(response, "usage", None):
        usage = {"input_tokens": getattr(response.usage, "input_tokens", None),
                 "output_tokens": getattr(response.usage, "output_tokens", None)}
    return LLMResponse(text=text, provider="claude", model=model,
                       data=_maybe_json(text) if schema else None, usage=usage)


# ------------------------------------------------------------------- ollama

def _ollama_reachable(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def generate_ollama(prompt: str, *, system: str, schema: dict | None = None,
                    model: str | None = None, max_tokens: int = 16000) -> LLMResponse:
    model = model or OLLAMA_MODEL
    payload: dict = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "options": {"num_predict": max_tokens},
    }
    if schema:
        payload["format"] = schema       # ollama takes a JSON schema directly

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LLMUnavailable(
            f"could not reach Ollama at {OLLAMA_HOST} ({exc}).\n"
            "  Start it with `ollama serve`, then pull a model:\n"
            f"  ollama pull {model}"
        ) from exc

    text = (body.get("message") or {}).get("content", "")
    return LLMResponse(text=text, provider="ollama", model=model,
                       data=_maybe_json(text) if schema else None,
                       usage={"eval_count": body.get("eval_count")})


# --------------------------------------------------------------------- stub

def generate_stub(prompt: str, *, system: str, schema: dict | None = None,
                  model: str | None = None, max_tokens: int = 16000) -> LLMResponse:
    """No model. Returns an empty payload so the caller falls back to a skeleton.

    Deliberately not a fake-quality script: a placeholder that reads like real
    writing is worse than an obvious blank, because it gets published by mistake.
    """
    return LLMResponse(text="", provider="stub", model="stub", data=None)


PROVIDERS = {"claude": generate_claude, "ollama": generate_ollama, "stub": generate_stub}


def available_provider(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import anthropic  # noqa: F401,PLC0415
        if _anthropic_credentials_present():
            return "claude"
    except ImportError:
        pass
    if _ollama_reachable():
        return "ollama"
    return "stub"


def describe_providers() -> list[tuple[str, bool, str]]:
    """(name, ready, note) for `reelforge doctor`."""
    rows = []
    try:
        import anthropic  # noqa: F401,PLC0415
        has_sdk = True
    except ImportError:
        has_sdk = False
    if not has_sdk:
        rows.append(("claude", False, "pip install anthropic"))
    elif not _anthropic_credentials_present():
        rows.append(("claude", False, "set ANTHROPIC_API_KEY (or run `ant auth login`)"))
    else:
        rows.append(("claude", True, CLAUDE_MODEL))
    reachable = _ollama_reachable()
    rows.append(("ollama", reachable,
                 f"{OLLAMA_MODEL} at {OLLAMA_HOST}" if reachable else f"not running at {OLLAMA_HOST}"))
    return rows


def generate(prompt: str, *, system: str, schema: dict | None = None,
             provider: str = "auto", model: str | None = None,
             max_tokens: int = 16000) -> LLMResponse:
    name = available_provider(provider)
    handler = PROVIDERS.get(name)
    if handler is None:
        raise LLMError(f"unknown provider '{name}' (choose from {', '.join(PROVIDERS)})")
    return handler(prompt, system=system, schema=schema, model=model, max_tokens=max_tokens)


def _maybe_json(text: str) -> dict | None:
    """Parse a JSON object out of a response, tolerating code fences."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```", 2)[1]
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
