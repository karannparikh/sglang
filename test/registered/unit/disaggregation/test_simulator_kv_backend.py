import ctypes
import inspect
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import numpy as np

from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import AbortNotification
from sglang.srt.disaggregation.fake.conn import FakeKVManager
from sglang.srt.disaggregation.simulator import conn as simulator_conn
from sglang.srt.disaggregation.simulator.conn import (
    SimulatorKVBootstrapServer,
    SimulatorKVManager,
    SimulatorKVReceiver,
    SimulatorKVSender,
    SimulatorRegistration,
    SimulatorTransferInfo,
    SimulatorTransferTask,
    compute_logical_transfer_bytes,
    modeled_transfer_latency_s,
)
from sglang.srt.disaggregation.utils import (
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _kv_args() -> KVArgs:
    args = KVArgs()
    args.kv_item_lens = [64, 64]
    args.aux_item_lens = [5, 7]
    args.aux_data_ptrs = [0, 0]
    args.state_item_lens = [[8, 4], [16]]
    args.page_size = 4
    return args


def _bare_manager() -> SimulatorKVManager:
    manager = object.__new__(SimulatorKVManager)
    manager.kv_args = _kv_args()
    manager.decode_kv_args_table = {}
    manager._decode_aux_indices = {}
    manager.transfer_infos = {}
    manager.transfer_lock = threading.Lock()
    manager.req_to_decode_prefix_len = {}
    manager.request_status = {}
    manager._kv_replica_factor = 1
    manager.is_mla_backend = False
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager._send_records = None
    manager.deferred_bootstrap = None
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.transfer_stats = {}
    manager._transfer_condition = threading.Condition()
    manager._transfer_heap = []
    manager._room_deadlines = {}
    manager._next_transfer_sequence = 0
    manager.enable_deferred_decode_kv_release = False
    manager.bandwidth_gb_per_s = 1.0
    manager.fixed_latency_ms = 2.0
    manager.pp_group = SimpleNamespace(is_last_rank=True)
    manager.send_kv_status_message = MagicMock()
    return manager


class TestSimulatorBackendSelection(CustomTestCase):
    def test_simulator_is_distinct_from_fake_warmup_backend(self):
        self.assertIs(
            get_kv_class(TransferBackend.SIMULATOR, KVClassType.MANAGER),
            SimulatorKVManager,
        )
        self.assertIs(
            get_kv_class(TransferBackend.SIMULATOR, KVClassType.SENDER),
            SimulatorKVSender,
        )
        self.assertIs(
            get_kv_class(TransferBackend.SIMULATOR, KVClassType.RECEIVER),
            SimulatorKVReceiver,
        )
        self.assertIs(
            get_kv_class(TransferBackend.SIMULATOR, KVClassType.BOOTSTRAP_SERVER),
            SimulatorKVBootstrapServer,
        )
        self.assertIs(
            get_kv_class(TransferBackend.FAKE, KVClassType.MANAGER),
            FakeKVManager,
        )

    def test_simulator_config_rejects_invalid_link_values(self):
        for field, value, message in (
            (
                "disaggregation_simulator_bandwidth_gb_per_s",
                0,
                "bandwidth-gb-per-s must be positive",
            ),
            (
                "disaggregation_simulator_fixed_latency_ms",
                -1,
                "fixed-latency-ms must be non-negative",
            ),
        ):
            with self.subTest(field=field):
                args = ServerArgs(
                    model_path="dummy",
                    disaggregation_transfer_backend="simulator",
                    **{field: value},
                )
                with self.assertRaisesRegex(ValueError, message):
                    handle_pd_disaggregation(args)


class TestSimulatorTransferModel(CustomTestCase):
    def test_logical_bytes_use_tokens_and_per_component_state_geometry(self):
        logical_bytes = compute_logical_transfer_bytes(
            kv_args=_kv_args(),
            kv_indices=np.array([3, 7], dtype=np.int32),
            index_slice=slice(0, 2),
            total_kv_pages=2,
            state_indices=[
                np.array([1, 2, 3], dtype=np.int32),
                None,
            ],
            aux_index=4,
            num_kv_tokens=5,
            include_aux=True,
        )

        self.assertEqual(logical_bytes, 5 * 32 + 3 * 12 + 12)

    def test_logical_bytes_apply_partial_page_only_to_its_shard(self):
        kwargs = dict(
            kv_args=_kv_args(),
            state_indices=None,
            aux_index=None,
            num_kv_tokens=5,
            include_aux=False,
            total_kv_pages=2,
        )

        full_page_bytes = compute_logical_transfer_bytes(
            kv_indices=np.array([3], dtype=np.int32),
            index_slice=np.array([0], dtype=np.int64),
            **kwargs,
        )
        partial_page_bytes = compute_logical_transfer_bytes(
            kv_indices=np.array([7], dtype=np.int32),
            index_slice=np.array([1], dtype=np.int64),
            **kwargs,
        )

        self.assertEqual(full_page_bytes, 4 * 32)
        self.assertEqual(partial_page_bytes, 32)

    def test_modeled_latency_decreases_with_bandwidth(self):
        slow = modeled_transfer_latency_s(
            logical_bytes=1_000_000_000,
            bandwidth_gb_per_s=1,
            fixed_latency_ms=2,
        )
        fast = modeled_transfer_latency_s(
            logical_bytes=1_000_000_000,
            bandwidth_gb_per_s=100,
            fixed_latency_ms=2,
        )

        self.assertAlmostEqual(slow, 1.002)
        self.assertAlmostEqual(fast, 0.012)
        self.assertGreater(slow, fast * 10)

    def test_metadata_size_does_not_scale_with_logical_payload(self):
        manager = _bare_manager()
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 1234
        manager.attn_tp_size = 1
        manager.attn_tp_rank = 0
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.kv_args.kv_item_lens = [1 << 40]
        manager.kv_args.aux_item_lens = [1 << 20]
        manager.kv_args.state_item_lens = [[1 << 35]]

        registration = manager.build_receiver_registration_message()
        transfer = manager.build_receiver_metadata_message(
            bootstrap_room=17,
            required_dst_info_num=1,
            kv_indices=np.array([1, 2, 3], dtype=np.int32),
            aux_index=0,
            state_indices=[np.array([4, 5], dtype=np.int32)],
            decode_prefix_len=64,
            is_dummy=False,
        )

        self.assertLess(sum(map(len, registration)), 1024)
        self.assertLess(sum(map(len, transfer)), 1024)
        source = inspect.getsource(simulator_conn)
        for forbidden in ("torch.empty", "from_address", "bytearray("):
            self.assertNotIn(forbidden, source)


class TestSimulatorControlPlane(CustomTestCase):
    def test_registration_and_transfer_metadata_reach_waiting_for_input(self):
        manager = _bare_manager()
        room = 41
        registration = SimulatorRegistration(
            peer_id="decode-0",
            endpoint="127.0.0.1",
            dst_port=8001,
            kv_item_lens=[64, 64],
            aux_item_lens=[5, 7],
            state_item_lens=[[8, 4], [16]],
            page_size=4,
            attn_tp_size=1,
            attn_tp_rank=0,
            dcp_size=1,
            dcp_rank=0,
        )
        transfer = SimulatorTransferInfo(
            room=room,
            peer_id="decode-0",
            endpoint="127.0.0.1",
            dst_port=8001,
            required_dst_info_num=1,
            num_kv_indices=3,
            state_index_counts=[2, None],
            aux_index=0,
            decode_prefix_len=8,
            is_dummy=False,
        )

        manager._handle_registration_message(
            [
                simulator_conn._REGISTER_TAG,
                msgspec.msgpack.encode(registration),
            ]
        )
        manager._handle_transfer_message(
            [
                simulator_conn._TRANSFER_TAG,
                msgspec.msgpack.encode(transfer),
            ]
        )

        self.assertEqual(manager.check_status(room), KVPoll.WaitingForInput)
        self.assertEqual(manager.req_to_decode_prefix_len[room], 8)
        self.assertEqual(manager.transfer_infos[room]["decode-0"], transfer)

    def test_zero_page_last_chunk_completes_and_records_fixed_latency(self):
        manager = _bare_manager()
        room = 42
        manager.request_status[room] = KVPoll.WaitingForInput
        manager.transfer_infos[room] = {
            "decode-0": SimulatorTransferInfo(
                room=room,
                peer_id="decode-0",
                endpoint="127.0.0.1",
                dst_port=8001,
                required_dst_info_num=1,
                num_kv_indices=0,
                state_index_counts=[],
                aux_index=None,
                decode_prefix_len=None,
                is_dummy=False,
            )
        }
        manager.decode_kv_args_table["decode-0"] = SimulatorRegistration(
            peer_id="decode-0",
            endpoint="127.0.0.1",
            dst_port=8001,
            kv_item_lens=[64, 64],
            aux_item_lens=[],
            state_item_lens=[[8, 4], [16]],
            page_size=4,
            attn_tp_size=1,
            attn_tp_rank=0,
            dcp_size=1,
            dcp_rank=0,
        )
        manager.kv_args.aux_item_lens = []
        manager.kv_args.aux_data_ptrs = []

        manager.add_transfer_request(
            bootstrap_room=room,
            kv_indices=np.array([3], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            aux_index=None,
        )
        manager.add_transfer_request(
            bootstrap_room=room,
            kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(1, 1),
            is_last_chunk=True,
            aux_index=None,
        )
        _, _, first_task = manager._transfer_heap.pop(0)
        manager._complete_transfer(first_task)
        self.assertEqual(manager.check_status(room), KVPoll.Transferring)
        _, _, last_task = manager._transfer_heap.pop()
        manager._complete_transfer(last_task)

        self.assertEqual(manager.check_status(room), KVPoll.Success)
        stats = manager.transfer_stats[room]
        self.assertEqual(stats.logical_bytes, 128)
        self.assertEqual(stats.chunks, 2)
        self.assertAlmostEqual(stats.modeled_latency_s, 0.004000128)
        self.assertIs(manager.pop_transfer_stats(room), stats)
        self.assertNotIn(room, manager.transfer_stats)

    def test_auxiliary_metadata_values_cross_without_kv_payload(self):
        room = 45
        src = _bare_manager()
        src.kv_args.aux_item_lens = [4]
        src_buffer = (ctypes.c_ubyte * 8)(0, 0, 0, 0, 9, 8, 7, 6)
        src.kv_args.aux_data_ptrs = [ctypes.addressof(src_buffer)]
        sent = []
        src._send_multipart_locked = lambda address, message, is_ipv6: sent.append(
            message
        )
        task = SimulatorTransferTask(
            room=room,
            is_last_chunk=True,
            targets=[("127.0.0.1", 8001)],
            target_aux_item_lens=[[4]],
            logical_bytes=4,
            modeled_latency_s=0.002,
            aux_index=1,
            send_record=object(),
        )

        src._send_aux_metadata(task)

        dst = _bare_manager()
        dst.kv_args.aux_item_lens = [4]
        dst_buffer = (ctypes.c_ubyte * 4)()
        dst.kv_args.aux_data_ptrs = [ctypes.addressof(dst_buffer)]
        dst._decode_aux_indices[room] = 0
        dst._handle_aux_data_message(sent[0])
        self.assertEqual(bytes(dst_buffer), bytes([9, 8, 7, 6]))

    def test_abort_waits_for_modeled_transfer_to_quiesce(self):
        manager = _bare_manager()
        room = 43
        manager.enable_deferred_decode_kv_release = True
        manager.request_status[room] = KVPoll.Transferring
        manager._staging_outstanding[room] = 1
        sent = []
        manager._send_abort_ack = lambda *args: sent.append(args)
        notification = AbortNotification(
            room=room,
            decode_ip="127.0.0.1",
            decode_port=9001,
            generation=3,
        )

        manager._handle_abort_message(notification.to_zmq())

        self.assertEqual(manager.check_status(room), KVPoll.Failed)
        self.assertEqual(sent, [])
        manager._mark_transfer_quiescent(room)
        self.assertEqual(len(sent), 1)
        self.assertNotIn(room, manager._staging_outstanding)

    def test_teardown_cancellation_releases_pending_send(self):
        manager = _bare_manager()
        room = 46
        manager.enable_deferred_decode_kv_release = True
        manager.request_status[room] = KVPoll.WaitingForInput
        manager.transfer_infos[room] = {
            "decode-0": SimulatorTransferInfo(
                room=room,
                peer_id="decode-0",
                endpoint="127.0.0.1",
                dst_port=8001,
                required_dst_info_num=1,
                num_kv_indices=1,
                state_index_counts=[],
                aux_index=None,
                decode_prefix_len=None,
                is_dummy=False,
            )
        }
        manager.decode_kv_args_table["decode-0"] = SimulatorRegistration(
            peer_id="decode-0",
            endpoint="127.0.0.1",
            dst_port=8001,
            kv_item_lens=[64, 64],
            aux_item_lens=[],
            state_item_lens=[[8, 4], [16]],
            page_size=4,
            attn_tp_size=1,
            attn_tp_rank=0,
            dcp_size=1,
            dcp_rank=0,
        )
        manager.kv_args.aux_item_lens = []
        manager.kv_args.aux_data_ptrs = []

        manager.add_transfer_request(
            bootstrap_room=room,
            kv_indices=np.array([3], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
        )
        task = manager._transfer_heap[0][2]
        self.assertEqual(task.send_record.finished, 0)

        manager._cancel_pending_transfers()

        self.assertEqual(manager.check_status(room), KVPoll.Failed)
        self.assertEqual(task.send_record.finished, 1)
        self.assertEqual(manager._transfer_heap, [])
        self.assertNotIn(room, manager._staging_outstanding)

    def test_receiver_timeout_fails_and_cleans_cached_peer(self):
        manager = _bare_manager()
        room = 44
        manager.waiting_timeout = 0
        manager.request_status[room] = KVPoll.WaitingForInput
        receiver = SimulatorKVReceiver.__new__(SimulatorKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = room
        receiver.bootstrap_addr = "prefill:8998"
        receiver.init_time = 1.0
        receiver.abort_notified = True
        receiver.conclude_state = None
        receiver.invalidate_cached_bootstrap_infos = MagicMock()

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        receiver.invalidate_cached_bootstrap_infos.assert_called_once_with()
        self.assertEqual(manager.check_status(room), KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
