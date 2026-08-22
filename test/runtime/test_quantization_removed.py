from types import SimpleNamespace

import pytest

from fluxserve.cli import _reject_unsupported_quantization, build_parser


@pytest.mark.parametrize(
    "quantization_config",
    [
        {"quant_method": "fp8"},
        {"quant_method": "modelopt", "quant_algo": "FP8"},
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        {"quantization": {"quant_algo": "FP8"}},
        {"quantization": {"quant_algo": "NVFP4"}},
    ],
)
def test_rejects_fp8_and_fp4_checkpoints(quantization_config):
    model_config = SimpleNamespace(quantization_config=quantization_config)

    with pytest.raises(ValueError, match="does not currently support FP8 or FP4"):
        _reject_unsupported_quantization(model_config)


@pytest.mark.parametrize("quantization_config", [None, {}, {"quant_method": "int8"}])
def test_allows_unquantized_metadata(quantization_config):
    model_config = SimpleNamespace(quantization_config=quantization_config)

    _reject_unsupported_quantization(model_config)


def test_quantization_flags_are_removed():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--model", "test", "--quantization", "fp8"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["bench_offline", "--model", "test", "--dataset", "test", "--use-quant"]
        )
