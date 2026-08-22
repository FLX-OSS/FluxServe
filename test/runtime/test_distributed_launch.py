import signal
from unittest.mock import Mock, call, patch

import pytest

from fluxserve.backend.distributed import launch


class FakeProcess:
    def __init__(self, *, alive=True, survives_terminate=False, pid=1234):
        self.alive = alive
        self.survives_terminate = survives_terminate
        self.pid = pid
        self.terminate = Mock(side_effect=self._terminate)
        self.kill = Mock(side_effect=self._kill)
        self.join = Mock()
        self.close = Mock()

    def _terminate(self):
        if not self.survives_terminate:
            self.alive = False

    def _kill(self):
        self.alive = False

    def is_alive(self):
        return self.alive


class FakeProcessContext:
    def __init__(self, processes, join_results):
        self.processes = processes
        self.join_results = iter(join_results)
        self.join = Mock(side_effect=lambda **_kwargs: next(self.join_results))


def _supervise_with_signals(context, signals, monotonic_values=None):
    installed_handlers = {}

    def install(signum, handler):
        if callable(handler):
            installed_handlers[signum] = handler
            return f"previous-{signum}"
        return None

    def join(**_kwargs):
        if signals:
            signum = signals.pop(0)
            installed_handlers[signum](signum, None)
        return next(context.join_results)

    context.join.side_effect = join
    monotonic = Mock(side_effect=monotonic_values or [0.0, 0.0])
    with patch.object(launch.signal, "signal", side_effect=install) as signal_mock, patch.object(
        launch.time, "monotonic", monotonic
    ):
        launch._supervise_processes(context)
    return signal_mock


def test_supervisor_leaves_completed_processes_alone():
    process = FakeProcess(alive=False)
    context = FakeProcessContext([process], [True])

    signal_mock = _supervise_with_signals(context, [])

    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    assert call(signal.SIGINT, f"previous-{signal.SIGINT}") in signal_mock.call_args_list
    assert call(signal.SIGTERM, f"previous-{signal.SIGTERM}") in signal_mock.call_args_list


def test_first_sigint_allows_workers_to_exit_gracefully():
    process = FakeProcess(alive=False)
    context = FakeProcessContext([process], [False, True])

    _supervise_with_signals(context, [signal.SIGINT], [10.0, 11.0])

    assert context.join.call_count == 3
    process.terminate.assert_not_called()
    process.close.assert_called_once_with()


def test_shutdown_timeout_terminates_remaining_workers():
    process = FakeProcess()
    context = FakeProcessContext([process], [False])

    _supervise_with_signals(context, [signal.SIGINT], [10.0, 13.0])

    process.terminate.assert_called_once_with()
    assert process.join.call_args_list == [
        call(timeout=launch._TERMINATE_GRACE_PERIOD_S),
        call(),
    ]
    process.kill.assert_not_called()
    process.close.assert_called_once_with()


def test_sigterm_is_forwarded_to_rank_zero():
    process = FakeProcess(alive=False, pid=4321)
    context = FakeProcessContext([process], [False, True])

    with patch.object(launch.os, "kill") as kill_mock, pytest.raises(
        SystemExit, match="143"
    ):
        _supervise_with_signals(context, [signal.SIGTERM], [10.0, 11.0])

    kill_mock.assert_not_called()


def test_sigterm_is_forwarded_when_rank_zero_is_alive():
    process = FakeProcess(pid=4321)
    context = FakeProcessContext([process], [False, True])

    with patch.object(launch.os, "kill") as kill_mock, pytest.raises(
        SystemExit, match="143"
    ):
        _supervise_with_signals(context, [signal.SIGTERM], [10.0, 11.0])

    kill_mock.assert_called_once_with(4321, signal.SIGTERM)


def test_second_signal_escalates_immediately():
    process = FakeProcess()
    context = FakeProcessContext([process], [False, False])

    _supervise_with_signals(
        context,
        [signal.SIGINT, signal.SIGINT],
        [10.0, 10.1],
    )

    assert context.join.call_count == 3
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with()
    process.close.assert_called_once_with()


def test_worker_that_survives_terminate_is_killed():
    process = FakeProcess(survives_terminate=True)

    launch._stop_remaining_processes([process])

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.join.call_args_list == [
        call(timeout=launch._TERMINATE_GRACE_PERIOD_S),
        call(),
    ]


def test_unexpected_process_context_failure_propagates():
    process = FakeProcess()
    context = FakeProcessContext([process], [])
    context.join.side_effect = RuntimeError("worker failed")

    with patch.object(launch.signal, "signal", return_value=signal.SIG_DFL):
        with pytest.raises(RuntimeError, match="worker failed"):
            launch._supervise_processes(context)

    process.terminate.assert_called_once_with()


def test_nonzero_worker_ignores_sigint_before_running_worker():
    worker = Mock()

    with patch.object(launch.signal, "signal") as signal_mock, patch.dict(
        launch.os.environ, {}, clear=True
    ):
        launch._local_worker_entry(1, 2, "file:///store", worker, "args")

    signal_mock.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
    worker.assert_called_once_with("args", init_method="file:///store")
