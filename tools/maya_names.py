"""Resolve public TINTERO_* settings and keep the existing MAYA_* and STRATA_* readers working."""
from __future__ import annotations

import os

OLD, NEW = "STRATA_", "MAYA_"
PUBLIC = "TINTERO_"
PUBLIC_SUFFIXES = frozenset("""
    API_KEY DRAIN_S ENGINE_QUIT_S ENGINE_STALL_S ENGINE_WRAP_S FAIR_SLICE_S HTTP_BACKLOG
    LOOP_GUARD LOOP_GUARD_COLLAPSE_MAX MAX_BODY_MIB MODEL_MANAGEMENT PROMPT_MEMO QUEUE_PING_S
    REQUEST_QUIET_S REQUEST_TIMEOUT_S SERVER_LOG UPDATE_CHECK
""".split()) | {
    "GLM_RESPONSES_" + suffix for suffix in """
        CACHE_ENTRIES CACHE_BYTES CACHE_TTL_S ENTRY_BYTES OWNER_ENTRIES OWNER_BYTES ASSEMBLY_BYTES
        JOB_WORKERS JOB_ENTRIES JOB_OWNER_ENTRIES JOB_TIMEOUT_S JOB_TTL_S JOB_INPUT_BYTES
        EVENT_BYTES EVENT_COUNT STREAM_READERS STREAM_OWNER_READERS STREAM_TIMEOUT_S
    """.split()
}


def shown(name: str) -> str:
    """The public spelling used in settings written to disk and shown to users."""
    key = read_as(name)
    suffix = key.removeprefix(NEW)
    return PUBLIC + suffix if key.startswith(NEW) and suffix in PUBLIC_SUFFIXES else key


def read_as(name: str) -> str:
    """The spelling used by existing Python and child process readers."""
    if name.startswith(OLD):
        return NEW + name[len(OLD):]
    if name.startswith(PUBLIC) and name[len(PUBLIC):] in PUBLIC_SUFFIXES:
        return NEW + name[len(PUBLIC):]
    return name


def alias_env(env) -> None:
    """Populate the existing runtime keys in place, including when the winning value is empty."""
    for k, v in list(env.items()):
        key = read_as(k)
        if (k.startswith(OLD) and key not in env) or (k.startswith(PUBLIC) and key != k):
            env[key] = v


def both_names(name: str) -> tuple[str, str]:
    """(MAYA_X, STRATA_X) for a setting under either name; a name with neither prefix is its own twice."""
    m = read_as(name)
    return m, (OLD + m[len(NEW):] if m.startswith(NEW) else m)


def lookup(env, name: str, default=None):
    """Read a setting in order of preference, preserving an empty value."""
    for k in (shown(name), *both_names(name)):
        v = env.get(k)
        if v is not None:
            return v
    return default


def export_env(target, env) -> None:
    """Apply a config to an environment under the spellings its readers accept."""
    for k, v in _normalized_env({str(k): v for k, v in (env or {}).items()}).items():
        for name in {shown(k), *both_names(k)}:
            target[name] = str(v)


def _normalized_env(env):
    out = {}
    for name, value in env.items():
        key = read_as(name)
        out[key] = env.get(shown(key), env.get(key, value))
    return out


def env_in(cfg):
    """Normalize a config's env keys in place for existing readers."""
    e = cfg.get("env") if isinstance(cfg, dict) else None
    if isinstance(e, dict):
        cfg["env"] = _normalized_env(e)
    return cfg


def env_out(cfg):
    """A config about to be written -> a copy with its "env" under the names users see."""
    e = cfg.get("env") if isinstance(cfg, dict) else None
    if not isinstance(e, dict):
        return cfg
    return {**cfg, "env": {shown(k): v for k, v in _normalized_env(e).items()}}


alias_env(os.environ)
