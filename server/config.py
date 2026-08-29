"""Server configuration: the OpenAI-facing model names and what they map to."""

import os

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# Origins allowed to call this API from a browser page, comma-separated, e.g.
# "https://app.example.com". Empty (the default) sends no CORS headers at all.
# "*" allows any site — convenient for testing, but note this server has no
# auth, so any page the user visits could then drive their DeepSeek account.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# Public model ids the server advertises (via /v1/models) and accepts, mapped
# to DeepSeek's `model_type` wire value plus whether DeepThink is on. Thinking
# is really a per-request tool (the `thinking` extra-body flag still works and
# ORs with this), but most OpenAI-compatible frontends (Zed, etc.) can only
# vary the model name — so the "-reasoner" ids bake it in, mirroring the
# official API's deepseek-chat / deepseek-reasoner naming.
MODEL_MAP = {
    # Instant — the fast default model
    "deepseek-chat":            {"model_type": "default", "thinking": False},
    # Expert — the stronger, slower model
    "deepseek-expert":          {"model_type": "expert",  "thinking": False},
    # Same models with DeepThink reasoning enabled
    "deepseek-reasoner":        {"model_type": "default", "thinking": True},
    "deepseek-expert-reasoner": {"model_type": "expert",  "thinking": True},
}

DEFAULT_MODEL = "deepseek-chat"


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 unknown models)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]["model_type"]


def model_thinking(name: str) -> bool:
    """Whether the model id has DeepThink reasoning baked in."""
    return MODEL_MAP[name]["thinking"]


# Extra names accepted for a model, for frontends with a fixed model list of
# their own: "MODEL_ALIASES=deepseek-v4-pro=deepseek-expert,gpt-4o=deepseek-chat".
MODEL_ALIASES = {}
for _pair in os.getenv("MODEL_ALIASES", "").split(","):
    if "=" in _pair:
        _alias, _target = _pair.split("=", 1)
        MODEL_ALIASES[_alias.strip()] = _target.strip()


def resolve_alias(name: str) -> str:
    """Map a caller-supplied alias to a real model id, or pass `name` through."""
    return MODEL_ALIASES.get(name, name)
