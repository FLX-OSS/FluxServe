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

"""Multi-EOS plumbing for the LLaDA decoders (LLaDA2.2 declares two stop
tokens, [156892, 156900], in its generation_config.json)."""

import json
import unittest

from fluxserve.backend.execution.decoders.factory import load_decoder
from fluxserve.backend.execution.decoders.utils import (
    normalize_eos_ids,
    resolve_checkpoint_eos_ids,
)
from fluxserve.backend.execution.forward_batch_info import RunnerConfig


class TestNormalizeEosIds(unittest.TestCase):
    def test_int_becomes_tuple(self):
        self.assertEqual(normalize_eos_ids(156892), (156892,))

    def test_dedup_preserves_order(self):
        self.assertEqual(
            normalize_eos_ids([156892, 156900, 156892]), (156892, 156900)
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            normalize_eos_ids([])


class TestFactoryEosIds(unittest.TestCase):
    DECODER_NAMES = ("threshold", "joint_threshold", "levenshtein_joint", "hierarchy")

    def test_default_is_single_eos(self):
        for name in self.DECODER_NAMES:
            decoder = load_decoder(
                RunnerConfig(parallel_decoding=name, threshold=0.7)
            )
            self.assertEqual(decoder.eos_ids, (156892,), name)
            self.assertEqual(decoder.eos_id, 156892, name)

    def test_config_eos_ids_reach_every_decoder(self):
        for name in self.DECODER_NAMES:
            decoder = load_decoder(
                RunnerConfig(
                    parallel_decoding=name,
                    threshold=0.7,
                    eos_ids=(156892, 156900),
                )
            )
            self.assertEqual(decoder.eos_ids, (156892, 156900), name)
            self.assertEqual(decoder.eos_id, 156892, name)

    def test_primary_eos_id_leads_and_dedups(self):
        decoder = load_decoder(
            RunnerConfig(
                parallel_decoding="threshold",
                eos_ids=(156900, 156892),
            )
        )
        self.assertEqual(decoder.eos_ids, (156892, 156900))
        self.assertEqual(decoder.eos_id, 156892)

    def test_use_credit_variant(self):
        decoder = load_decoder(
            RunnerConfig(
                parallel_decoding="threshold",
                use_credit=True,
                eos_ids=(156892, 156900),
            )
        )
        self.assertEqual(decoder.eos_ids, (156892, 156900))


class TestResolveCheckpointEosIds(unittest.TestCase):
    def _checkpoint_dir(self, tmp_path, generation_config=None):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        if generation_config is not None:
            (tmp_path / "generation_config.json").write_text(
                json.dumps(generation_config)
            )
        return str(tmp_path)

    def test_list_of_stop_tokens(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint_dir(
                Path(tmp), {"eos_token_id": [156892, 156900]}
            )
            self.assertEqual(resolve_checkpoint_eos_ids(path), (156892, 156900))

    def test_scalar_stop_token(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint_dir(Path(tmp), {"eos_token_id": 156892})
            self.assertEqual(resolve_checkpoint_eos_ids(path), (156892,))

    def test_missing_generation_config_is_empty(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint_dir(Path(tmp))
            self.assertEqual(resolve_checkpoint_eos_ids(path), ())

    def test_generation_config_without_eos_is_empty(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint_dir(Path(tmp), {"do_sample": False})
            self.assertEqual(resolve_checkpoint_eos_ids(path), ())


if __name__ == "__main__":
    unittest.main()
