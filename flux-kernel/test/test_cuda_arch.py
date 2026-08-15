from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from flux_kernel.cuda.build import (
    normalize_cuda_arch,
    nvcc_supported_arches,
    parse_cuda_arch_list,
    resolve_cuda_arches,
)


class CudaArchTest(unittest.TestCase):
    def test_normalizes_torch_and_nvcc_spellings(self):
        self.assertEqual(normalize_cuda_arch("10.0a"), "100a")
        self.assertEqual(normalize_cuda_arch("sm_100a"), "100a")
        self.assertEqual(normalize_cuda_arch("9.0+PTX"), "90")

    def test_parses_separators_and_removes_duplicates(self):
        self.assertEqual(
            parse_cuda_arch_list("9.0; 10.0a,sm_90 100a"),
            ("90", "100a"),
        )

    def test_flux_override_precedes_torch_override(self):
        arches = resolve_cuda_arches(
            ("90",),
            nvcc="nvcc",
            env={
                "FLUX_KERNEL_CUDA_ARCH": "100a",
                "TORCH_CUDA_ARCH_LIST": "9.0",
            },
            supported_arches=frozenset({"90", "100a"}),
        )
        self.assertEqual(arches, ("100a",))

    def test_detects_b200_as_architecture_specific_sm100(self):
        arches = resolve_cuda_arches(
            ("90",),
            nvcc="nvcc",
            env={},
            detected_capability=(10, 0),
            supported_arches=frozenset({"90", "100", "100a"}),
        )
        self.assertEqual(arches, ("100a",))

    def test_filters_unsupported_automatic_defaults(self):
        arches = resolve_cuda_arches(
            ("90", "100a", "120a"),
            nvcc="nvcc",
            env={},
            detected_capability=None,
            supported_arches=frozenset({"90", "100a"}),
        )
        self.assertEqual(arches, ("90", "100a"))

    def test_rejects_unsupported_explicit_architecture(self):
        with self.assertRaisesRegex(RuntimeError, "sm_100a"):
            resolve_cuda_arches(
                ("90",),
                nvcc="old-nvcc",
                env={"FLUX_KERNEL_CUDA_ARCH": "100a"},
                supported_arches=frozenset({"90"}),
            )

    @mock.patch("subprocess.check_output")
    def test_reads_nvcc_supported_architectures(self, check_output):
        check_output.return_value = "sm_80\nsm_90\nsm_100\n"
        self.assertEqual(
            nvcc_supported_arches("nvcc"),
            frozenset({"80", "90", "90a", "100", "100a"}),
        )
        check_output.assert_called_once_with(
            ["nvcc", "--list-gpu-code"],
            text=True,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    unittest.main()
