import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSkipDetokenizerInit(unittest.TestCase):
    def test_cli_flag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = ServerArgs.from_cli_args(
            parser.parse_args(
                ["--model-path", "dummy", "--skip-detokenizer-init"]
            )
        )

        self.assertTrue(args.skip_detokenizer_init)

    def test_engine_skips_detokenizer_processes(self):
        server_args = ServerArgs(
            model_path="dummy",
            skip_detokenizer_init=True,
        )
        with patch.object(
            Engine,
            "_launch_detokenizer_subprocesses",
        ) as launch:
            result = Engine._launch_detokenizer_subprocesses_if_enabled(
                server_args=server_args,
                port_args=MagicMock(),
                run_detokenizer_process_func=MagicMock(),
            )

        self.assertEqual(result, ([], []))
        launch.assert_not_called()

    def test_engine_launches_detokenizer_by_default(self):
        server_args = ServerArgs(model_path="dummy")
        expected = ([MagicMock()], ["detokenizer"])
        port_args = MagicMock()
        runner = MagicMock()
        with patch.object(
            Engine,
            "_launch_detokenizer_subprocesses",
            return_value=expected,
        ) as launch:
            result = Engine._launch_detokenizer_subprocesses_if_enabled(
                server_args=server_args,
                port_args=port_args,
                run_detokenizer_process_func=runner,
            )

        self.assertEqual(result, expected)
        launch.assert_called_once_with(
            server_args=server_args,
            port_args=port_args,
            run_detokenizer_process_func=runner,
        )

    @patch(
        "sglang.srt.managers.scheduler_components.ipc_channels.zmq.Context",
        return_value=MagicMock(),
    )
    @patch(
        "sglang.srt.managers.scheduler_components.ipc_channels.get_zmq_socket"
    )
    def test_scheduler_routes_output_around_detokenizer(
        self,
        get_zmq_socket,
        _context,
    ):
        get_zmq_socket.side_effect = (
            lambda _context, _socket_type, endpoint, _bind: endpoint
        )
        port_args = SimpleNamespace(
            scheduler_input_ipc_name="scheduler",
            rpc_ipc_name="rpc",
            tokenizer_ipc_name="tokenizer",
            detokenizer_ipc_name="detokenizer",
            metrics_ipc_name="metrics",
        )

        for skip_tokenizer_init, skip_detokenizer_init, expected in (
            (False, False, "detokenizer"),
            (True, False, "tokenizer"),
            (False, True, "tokenizer"),
        ):
            with self.subTest(
                skip_tokenizer_init=skip_tokenizer_init,
                skip_detokenizer_init=skip_detokenizer_init,
            ):
                channels = SchedulerIpcChannels.create(
                    port_args=port_args,
                    is_rank_zero=True,
                    skip_tokenizer_init=skip_tokenizer_init,
                    skip_detokenizer_init=skip_detokenizer_init,
                    metrics_enabled=False,
                    enable_scripted_runtime=False,
                )
                self.assertEqual(
                    channels.send_to_detokenizer.socket,
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
