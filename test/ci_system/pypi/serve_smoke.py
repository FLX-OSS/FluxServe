#!/usr/bin/env python3
"""Serving smoke test for a venv: launch the server from a CI task spec, wait for readiness, send one chat request.

The server runs in its own process group and is stopped by that group, never by pattern: the SM120 runner shares
its node with interactive work.
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # test/ci_system
from pipeline import ensure_ready_port_free, load_yaml, poll_readiness, start_server, stop_server  # noqa: E402

DEFAULT_SPEC = "test/ci/1N1G-SM120/eval/nemotron-diffusion-14b-evalscope-gsm8k.yaml"
PROMPT = "What is 12 * 7? Answer with just the number."


def served_model(command: str) -> str:
    argv = shlex.split(command)
    return argv[argv.index("--model") + 1]


def completion_problem(status: int, payload: dict):
    """Return (answer text, problem or None) for a /v1/chat/completions response."""
    message = payload["choices"][0]["message"]
    text = (message.get("content") or message.get("reasoning_content") or "").strip()
    # fluxserve doesn't report `usage`; check it only when a server does.
    tokens = (payload.get("usage") or {}).get("completion_tokens")
    if status != 200:
        return text, f"HTTP {status}"
    if not text:
        return text, "the completion is empty"
    if tokens is not None and tokens <= 0:
        return text, f"the server reported {tokens} completion tokens"
    return text, None


def annotate(message: str) -> None:
    """Surface a failure on the GitHub check page, where the job log needs admin rights to download."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        data = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title=serving smoke test::{data}", flush=True)


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(no server log)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC, help="CI task spec whose server section to launch")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True, help="Server log, PGID file and response")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    server = load_yaml(args.repo_root / args.spec)["server"]
    ready = server["ready"]
    command = server["command"]
    endpoint = urlparse(str(ready["url"]))._replace(path="/v1/chat/completions").geturl()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "server.log"
    pgid_file = args.out_dir / "server.pgid"

    ensure_ready_port_free(ready, dry_run=False)
    # The weights are in the shared cache; never reach out to the Hub from CI.
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    process = start_server(f"{command} > {shlex.quote(str(log))} 2>&1", env, args.repo_root, dry_run=False)
    pgid_file.write_text(str(process.pid))
    try:
        poll_readiness(ready, dry_run=False, process=process)
        body = {
            "model": served_model(command),
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        request = Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=600) as response:
            status = response.status
            payload = json.load(response)
        (args.out_dir / "smoke-response.json").write_text(json.dumps(payload, indent=1))
        text, problem = completion_problem(status, payload)
        print(f"HTTP {status}: {text[:200]!r}")
        if problem:
            raise RuntimeError(problem)
        if "84" not in text:
            print("note: the answer does not contain 84; serving works, but check the output")
        return 0
    except Exception as exc:
        annotate(f"{exc}")
        print(f"smoke test failed: {exc}\n--- last lines of {log}:\n{tail(log)}", file=sys.stderr)
        return 1
    finally:
        stop_server(process)
        pgid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
