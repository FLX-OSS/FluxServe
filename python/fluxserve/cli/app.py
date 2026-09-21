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
    Top-level FluxServe CLI parser and command dispatcher.
"""

from __future__ import annotations

from fluxserve.cli.bench_offline import bench_offline
from fluxserve.cli.common import configure_logging
from fluxserve.cli.launch import build_parser, launch


def main() -> None:
    """Parse the command line and dispatch the selected command."""
    configure_logging()
    args = build_parser().parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "env":
        from fluxserve.env import main as env_main

        env_main()
    elif args.command == "bench":
        args.dispatch_function(args)
    elif args.command == "bench_offline":
        bench_offline(args)
