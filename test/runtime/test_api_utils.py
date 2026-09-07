import io
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import fluxserve
import fluxserve.version
from fluxserve.backend.entrypoints import http_server
from fluxserve.backend.entrypoints.api_utils import log_startup_banner, resolved_config


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


def test_startup_banner_uses_color_for_tty(monkeypatch, capsys):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FLUXSERVE_DISABLE_LOG_LOGO", raising=False)
    stream = TTYBuffer()

    log_startup_banner("0.1", "org/model", "0.0.0.0", 8000, stream=stream)

    output = stream.getvalue()
    with capsys.disabled():
        print(output, end="")
    assert "\033[38;2;250;197;191m" in output
    assert "\033[95m" not in output
    assert "████████  ██                           ████████" in output
    assert "██      █████    ██      ██  ████████" in output
    assert "\U0001F9A9" in output
    assert "\n\nVersion  \U0001F9A9 0.1" in output
    assert "Model    org/model" in output
    assert "Server   http://0.0.0.0:8000" in output
    assert "`>" not in output


def test_startup_banner_is_monochrome_for_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = io.StringIO()

    log_startup_banner("0.1", "model", "localhost", 9000, stream=stream)

    assert "\033[" not in stream.getvalue()


def test_no_color_disables_color_for_tty(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    stream = TTYBuffer()

    log_startup_banner("0.1", "model", "localhost", 9000, stream=stream)

    assert "\033[" not in stream.getvalue()


def test_disabled_logo_keeps_metadata_and_config(monkeypatch):
    monkeypatch.setenv("FLUXSERVE_DISABLE_LOG_LOGO", "yes")
    stream = TTYBuffer()

    log_startup_banner("0.1", "model", "localhost", 9000, stream=stream)

    output = stream.getvalue()
    assert "████████" not in output
    assert "Version  \U0001F9A9 0.1" in output
    assert "Model    model" in output
    assert "Server   http://localhost:9000" in output
    assert 'Config   {' in output


@dataclass
class FakeServerConfig:
    model_name: str = "org/model"
    model_config: object = object()
    tp_size: int = 2
    capture_sizes: tuple[int, ...] = (1, 4)


@dataclass
class FakeRunnerConfig:
    attention_backend: str = "flashinfer"
    cuda_graph_log_callback: object = object()
    enable_decode_cuda_graph: bool = True


def test_resolved_config_serializes_final_values_and_excludes_runtime_objects():
    config = resolved_config(FakeServerConfig(), FakeRunnerConfig())

    assert config == {
        "runner": {
            "attention_backend": "flashinfer",
            "enable_decode_cuda_graph": True,
        },
        "server": {
            "capture_sizes": [1, 4],
            "model_name": "org/model",
            "tp_size": 2,
        },
    }


def test_http_run_logs_banner_once_before_uvicorn(monkeypatch, capsys):
    uvicorn_run = Mock()
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=uvicorn_run))
    monkeypatch.delenv("FLUXSERVE_DISABLE_LOG_LOGO", raising=False)
    server_config = FakeServerConfig()
    runner_config = FakeRunnerConfig()
    engine = SimpleNamespace(server_args=server_config)
    app = object()
    monkeypatch.setattr(http_server, "create_app", Mock(return_value=app))

    http_server.run(
        engine,
        host="127.0.0.1",
        port=8123,
        runner_config=runner_config,
    )

    output = capsys.readouterr().err
    first_wordmark_line = "████████  ██                           ████████"
    assert output.count(first_wordmark_line) == 1
    assert "\n\nVersion  \U0001F9A9 0.1" in output
    assert "Model    org/model" in output
    assert '"tp_size": 2' in output
    assert '"attention_backend": "flashinfer"' in output
    assert "model_config" not in output
    assert "cuda_graph_log_callback" not in output
    uvicorn_run.assert_called_once_with(
        app, host="127.0.0.1", port=8123, timeout_keep_alive=30
    )


def test_version_is_consistent():
    assert fluxserve.__version__ == "0.1"
    assert fluxserve.version.__version__ == "0.1"


if __name__ == "__main__":
    visual_stream = TTYBuffer()
    log_startup_banner(
        fluxserve.__version__,
        "org/model",
        "0.0.0.0",
        8000,
        stream=visual_stream,
    )
    print(visual_stream.getvalue(), end="")
