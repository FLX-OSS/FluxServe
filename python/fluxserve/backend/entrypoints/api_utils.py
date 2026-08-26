# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import TextIO


_WORDMARK = (
    "████████  ██                           ████████ ",
    "██        ██                           ██ ",
    "██        ██    ██     ██   ██    ██   ██         ██████    ██  ██  ██       ██   ██████",
    "████████  ██    ██     ██    ██  ██    ████████  ██    ██   ██ ██    ██     ██   ██    ██",
    "██        ██    ██     ██      ██            ██  ████████   ███       ██   ██    ████████",
    "██        ██    ██     ██    ██  ██          ██  ██         ██         ██ ██     ██ ",
    "██        ██      █████     ██    ██   ████████  ████████   ██          ███      ████████",
)




_CYAN = "\033[38;2;250;197;191m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_EXCLUDED_CONFIG_FIELDS = frozenset({"model_config", "cuda_graph_log_callback"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _supports_color(stream: TextIO) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    try:
        return stream.isatty()
    except (AttributeError, OSError):
        return False


def _serialize_config(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serialize_config(getattr(value, field.name))
            for field in fields(value)
            if field.name not in _EXCLUDED_CONFIG_FIELDS
        }
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_config(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def resolved_config(server_config, runner_config) -> dict:
    """Return a deterministic, JSON-compatible runtime configuration."""
    return {
        "runner": _serialize_config(runner_config),
        "server": _serialize_config(server_config),
    }


def log_startup_banner(
    version: str,
    model: str,
    host: str,
    port: int,
    *,
    server_config=None,
    runner_config=None,
    stream: TextIO | None = None,
) -> None:
    """Write the FluxServe startup banner to a terminal or log stream."""
    if stream is None:
        stream = sys.stderr
    server = f"http://{host}:{port}"

    color = _supports_color(stream)
    lines = []
    if not _env_flag("FLUXSERVE_DISABLE_LOG_LOGO"):
        for wordmark in _WORDMARK:
            if color:
                wordmark = f"{_BOLD}{_CYAN}{wordmark}{_RESET}"
            lines.append(wordmark.rstrip())
        lines.append("")

    lines.extend(
        (
            f"Version  \U0001F9A9 {version}",
            f"Model    {model}",
            f"Server   {server}",
        )
    )
    config_json = json.dumps(
        resolved_config(server_config, runner_config),
        indent=2,
        sort_keys=True,
    ).splitlines()
    lines.append(f"Config   {config_json[0]}")
    lines.extend(f"         {line}" for line in config_json[1:])

    stream.write("\n" + "\n".join(lines) + "\n")
    stream.flush()
