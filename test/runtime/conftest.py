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

"""Shared fixtures for the runtime suites."""

import importlib
import os
import shutil
import sys

import pytest

LLADA22_REPO_ID = "inclusionAI/LLaDA2.2-flash"
LLADA22_REFERENCE_FILES = ("configuration_llada2_moe.py", "modeling_llada2_moe.py")


def _has_reference_files(directory):
    return bool(directory) and all(
        os.path.isfile(os.path.join(directory, name))
        for name in LLADA22_REFERENCE_FILES
    )


def llada22_reference_dir():
    """Directory holding the LLaDA2.2 reference modeling code, or None.

    The parity suites check FluxServe's gate and decoder against the code that
    ships inside the ``inclusionAI/LLaDA2.2-flash`` checkpoint. Those files are
    not vendored, so they are resolved at run time:

    1. ``FLUXSERVE_LLADA22_REF_DIR``, if set -- a directory holding the two
       files below. Use it to point at a hand-placed copy.
    2. Otherwise the checkpoint's snapshot in the local Hugging Face cache,
       which is where a normal download puts them.

    Only those two small Python files are needed, not the 206GB of weights,
    and no network access is attempted.
    """
    override = os.environ.get("FLUXSERVE_LLADA22_REF_DIR")
    if override:
        return override if _has_reference_files(override) else None
    try:
        from huggingface_hub import snapshot_download

        cached = snapshot_download(
            LLADA22_REPO_ID,
            allow_patterns=list(LLADA22_REFERENCE_FILES),
            local_files_only=True,
        )
    except Exception:
        # Not cached, hub unavailable, or an incomplete snapshot. The suites
        # skip; a parity test must never fail for want of the reference.
        return None
    return cached if _has_reference_files(cached) else None


@pytest.fixture(scope="session")
def reference_module(tmp_path_factory):
    """The checkpoint's ``modeling_llada2_moe`` module, imported from a copy.

    The files are copied into a throwaway package so the checkpoint's own
    directory is never written to and the import cannot collide with anything
    else in ``sys.modules``.
    """
    source = llada22_reference_dir()
    if source is None:
        pytest.skip(
            "LLaDA2.2 reference modeling code not available; set "
            "FLUXSERVE_LLADA22_REF_DIR or cache the checkpoint locally"
        )

    # The checkpoint's modeling code targets transformers 5.x; the only import
    # missing from 4.57 is create_bidirectional_mask, which neither the gate
    # nor the decode loop uses. Shim it before loading.
    import transformers.masking_utils as mu

    if not hasattr(mu, "create_bidirectional_mask"):
        mu.create_bidirectional_mask = (
            lambda config=None, inputs_embeds=None, attention_mask=None, **kw: attention_mask
        )

    package = "llada22_reference_pkg"
    pkg_root = tmp_path_factory.mktemp(package)
    pkg = pkg_root / package
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    for name in LLADA22_REFERENCE_FILES:
        shutil.copy(os.path.join(source, name), pkg / name)
    sys.path.insert(0, str(pkg_root))
    try:
        yield importlib.import_module(f"{package}.modeling_llada2_moe")
    finally:
        sys.path.remove(str(pkg_root))
