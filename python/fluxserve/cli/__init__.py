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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
    Command parsing and runtime dispatch.
"""

from __future__ import annotations

import argparse

from fluxserve.cli.args import (
    add_bench_offline_subparser,
    add_bench_subparser,
    add_launch_subparser,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluxserve")
    sub = parser.add_subparsers(dest="command", required=True)
    add_launch_subparser(sub)
    add_bench_subparser(sub)
    add_bench_offline_subparser(sub)
    sub.add_parser("env", help="Print environment and dependency information.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "launch":
        from fluxserve.cli.launch import launch

        launch(args)
    elif args.command == "bench":
        import asyncio

        from fluxserve.cli.bench import run_serving_benchmark

        asyncio.run(run_serving_benchmark(args))
    elif args.command == "bench_offline":
        from fluxserve.cli.bench_offline import bench_offline

        bench_offline(args)
    elif args.command == "env":
        from fluxserve.cli.env import main as env_main

        env_main()

