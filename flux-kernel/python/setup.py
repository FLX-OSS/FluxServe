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

from __future__ import annotations

import os
from pathlib import Path
from setuptools import Command, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel


ROOT = Path(__file__).resolve().parent


class BuildNative(Command):
    description = "Build FluxServe native CUDA kernels"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        if os.environ.get("FLUX_KERNEL_SKIP_CUDA_BUILD"):
            print("FLUX_KERNEL_SKIP_CUDA_BUILD is set; skipping native CUDA build")
            return

        from flux_kernel.cuda.rmsnorm import ensure_rmsnorm_fused_parallel_built

        ensure_rmsnorm_fused_parallel_built(
            force=False,
            verbose=bool(getattr(self, "verbose", False)),
        )


class BuildPyWithNative(build_py):
    def run(self) -> None:
        self.run_command("build_native")
        super().run()


class DevelopWithNative(develop):
    def run(self) -> None:
        self.run_command("build_native")
        super().run()


class EditableWheelWithNative(editable_wheel):
    def run(self) -> None:
        self.run_command("build_native")
        super().run()


setup(
    name="flux-kernel",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "flux_kernel.cuda.rmsnorm": [
            "csrc/**/*.cu",
            "csrc/**/*.cuh",
            "csrc/**/*.h",
            "objs/*.so",
            "objs/*.build",
        ],
    },
    include_package_data=True,
    cmdclass={
        "build_native": BuildNative,
        "build_py": BuildPyWithNative,
        "develop": DevelopWithNative,
        "editable_wheel": EditableWheelWithNative,
    },
)
