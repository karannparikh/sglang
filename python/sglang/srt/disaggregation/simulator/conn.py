from __future__ import annotations

import heapq
import logging
import math
import threading
import time
from typing import List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
from sglang.srt.disaggregation.base.conn import (
    KVArgs,
    KVPoll,
    KVTransferMetric,
)
from sglang.srt.disaggregation.common.conn import (
    ABORT_TAG,
    AbortNotification,
    AckTarget,
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import AuxDataCodec
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.simulator_topology import (
    SimulatorLink,
    SimulatorTopology,
    load_simulator_topology,
    parse_simulator_location,
)
from sglang.srt.runtime_context import get_disagg
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

_REGISTER_TAG = b"SIMULATOR_REGISTER"
_TRANSFER_TAG = b"SIMULATOR_TRANSFER"
_AUX_DATA_TAG = b"SIMULATOR_AUX_DATA"
_STATUS_TAG = b"SIMULATOR_STATUS"
_BYTES_PER_GB = 1_000_000_000


class SimulatorRegistration(msgspec.Struct, frozen=True):
    peer_id: str
    endpoint: str
    dst_port: int
    kv_item_lens: List[int]
    aux_item_lens: List[int]
    state_item_lens: List[List[int]]
    page_size: int
    attn_tp_size: int
    attn_tp_rank: int
    dcp_size: int
    dcp_rank: int
    location: Optional[List[str]] = None


class SimulatorTransferInfo(msgspec.Struct, frozen=True):
    room: int
    peer_id: str
    endpoint: str
    dst_port: int
    required_dst_info_num: int
    num_kv_indices: int
    state_index_counts: List[Optional[int]]
    aux_index: Optional[int]
    decode_prefix_len: Optional[int]
    is_dummy: bool


class SimulatorTransferTask(msgspec.Struct, frozen=True):
    room: int
    is_last_chunk: bool
    targets: List[Tuple[str, int]]
    target_aux_item_lens: List[List[int]]
    logical_bytes: int
    modeled_latency_s: float
    link_name: Optional[str]
    aux_index: Optional[int]
    send_record: object


class SimulatorTransferStats(msgspec.Struct):
    logical_bytes: int = 0
    modeled_latency_s: float = 0.0
    chunks: int = 0


def compute_logical_transfer_bytes(
    *,
    kv_args: KVArgs,
    kv_indices: npt.NDArray[np.int32],
    index_slice,
    total_kv_pages: int,
    state_indices: Optional[List],
    aux_index: Optional[int],
    num_kv_tokens: Optional[int],
    include_aux: bool,
    kv_replica_factor: int = 1,
    target_aux_item_lens: Optional[List[List[int]]] = None,
) -> int:
    kv_item_bytes = sum(kv_args.kv_item_lens)
    if num_kv_tokens is None:
        kv_bytes = len(kv_indices) * kv_item_bytes
    else:
        if num_kv_tokens < 0:
            raise ValueError("num_kv_tokens must be non-negative")
        if kv_args.page_size <= 0 or kv_item_bytes % kv_args.page_size != 0:
            raise ValueError(
                "Simulator KV item bytes must divide evenly by the physical page size"
            )
        selected_tokens = len(kv_indices) * kv_args.page_size
        partial_page_tokens = num_kv_tokens % kv_args.page_size
        if partial_page_tokens and selected_tokens:
            final_page = total_kv_pages - 1
            if isinstance(index_slice, slice):
                start = index_slice.start or 0
                stop = index_slice.stop or start
                includes_final_page = start <= final_page < stop
            else:
                includes_final_page = bool(np.any(index_slice == final_page))
            if includes_final_page:
                selected_tokens -= kv_args.page_size - partial_page_tokens
        kv_bytes = selected_tokens * (kv_item_bytes // kv_args.page_size)

    state_bytes = 0
    if state_indices is not None:
        if len(state_indices) > len(kv_args.state_item_lens):
            raise ValueError(
                "Simulator state index components exceed registered state components"
            )
        for component, indices in enumerate(state_indices):
            if indices is not None:
                state_bytes += len(indices) * sum(kv_args.state_item_lens[component])

    aux_bytes = 0
    if include_aux and aux_index is not None:
        if target_aux_item_lens is None:
            aux_bytes = sum(kv_args.aux_item_lens)
        else:
            aux_bytes = sum(
                sum(kv_args.aux_item_lens[: len(remote_item_lens)])
                for remote_item_lens in target_aux_item_lens
            )
    return (kv_bytes + state_bytes) * kv_replica_factor + aux_bytes


def modeled_transfer_latency_s(
    *,
    logical_bytes: int,
    bandwidth_gb_per_s: float,
    fixed_latency_ms: float,
) -> float:
    if logical_bytes < 0:
        raise ValueError("logical_bytes must be non-negative")
    if not math.isfinite(bandwidth_gb_per_s) or bandwidth_gb_per_s <= 0:
        raise ValueError("bandwidth_gb_per_s must be finite and positive")
    if not math.isfinite(fixed_latency_ms) or fixed_latency_ms < 0:
        raise ValueError("fixed_latency_ms must be finite and non-negative")
    return fixed_latency_ms / 1000 + logical_bytes / (
        bandwidth_gb_per_s * _BYTES_PER_GB
    )


class SimulatorKVManager(CommonKVManager):
    supports_role_switch = True
    supports_deferred_decode_kv_release = True
    kv_status_msg_tag = _STATUS_TAG
    kv_status_msg_carries_reason = True

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        disagg = get_disagg()
        self.bandwidth_gb_per_s = disagg.disaggregation_simulator_bandwidth_gb_per_s
        self.fixed_latency_ms = disagg.disaggregation_simulator_fixed_latency_ms
        self.topology: Optional[SimulatorTopology] = load_simulator_topology(
            disagg.disaggregation_simulator_topology
        )
        self.location = (
            None
            if disagg.disaggregation_simulator_location is None
            else parse_simulator_location(disagg.disaggregation_simulator_location)
        )
        if self.topology is not None and self.location is not None:
            self.topology.validate_location(self.location)
        self.enable_staging = False
        self.decode_kv_args_table: dict[str, SimulatorRegistration] = {}
        self._decode_aux_indices: dict[int, int] = {}
        self.transfer_lock = threading.Lock()
        self.transfer_stats: dict[int, SimulatorTransferStats] = {}
        self._transfer_condition = threading.Condition()
        self._transfer_heap: list[tuple[float, int, SimulatorTransferTask]] = []
        self._room_deadlines: dict[int, float] = {}
        self._next_transfer_sequence = 0
        self._stopped = False
        self._worker_threads: List[threading.Thread] = []

        if disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_prefill_control_worker()
            worker = threading.Thread(
                target=self._transfer_worker,
                daemon=True,
                name=(f"simulator-xfer-dp{self.system_dp_rank}-tp{self.attn_tp_rank}"),
            )
            worker.start()
            self._worker_threads.append(worker)
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self._start_decode_status_worker()
            self._worker_threads.append(self._start_heartbeat_checker_thread())

    def prepare_send_state_indices(
        self, state_indices: Optional[List]
    ) -> Optional[List]:
        if self._should_skip_cp_replicated_state_transfer():
            return None
        return state_indices

    def build_receiver_registration_message(self) -> List[bytes]:
        registration = SimulatorRegistration(
            peer_id=self._peer_id(),
            endpoint=self.local_ip,
            dst_port=self.rank_port,
            kv_item_lens=list(self.kv_args.kv_item_lens),
            aux_item_lens=list(self.kv_args.aux_item_lens),
            state_item_lens=[
                list(component) for component in self.kv_args.state_item_lens
            ],
            page_size=self.kv_args.page_size,
            attn_tp_size=self.attn_tp_size,
            attn_tp_rank=self.attn_tp_rank,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            location=None if self.location is None else list(self.location),
        )
        return [_REGISTER_TAG, msgspec.msgpack.encode(registration)]

    def build_receiver_metadata_message(
        self,
        *,
        bootstrap_room: int,
        required_dst_info_num: int,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int],
        state_indices: Optional[List],
        decode_prefix_len: Optional[int],
        is_dummy: bool,
        device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
    ) -> List[bytes]:
        del device_kv_indices
        state_index_counts = []
        if not is_dummy and state_indices is not None:
            state_index_counts = [
                None if indices is None else len(indices) for indices in state_indices
            ]
        transfer_info = SimulatorTransferInfo(
            room=bootstrap_room,
            peer_id=self._peer_id(),
            endpoint=self.local_ip,
            dst_port=self.rank_port,
            required_dst_info_num=required_dst_info_num,
            num_kv_indices=0 if is_dummy else len(kv_indices),
            state_index_counts=state_index_counts,
            aux_index=None if is_dummy else aux_index,
            decode_prefix_len=decode_prefix_len,
            is_dummy=is_dummy,
        )
        if not is_dummy and aux_index is not None:
            self._decode_aux_indices[bootstrap_room] = aux_index
        return [_TRANSFER_TAG, msgspec.msgpack.encode(transfer_info)]

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        *,
        wait_event: Optional[object] = None,
        indices_event: Optional[object] = None,
        trace_ctx=None,
    ) -> None:
        del trace_ctx
        if wait_event is not None:
            wait_event.synchronize()
        if indices_event is not None:
            indices_event.synchronize()

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            return

        with self.transfer_lock:
            infos = self.transfer_infos.get(bootstrap_room)
            if not infos:
                return
            targets = []
            target_aux_item_lens = []
            target_locations = []
            for info in infos.values():
                if info.is_dummy:
                    continue
                targets.append((info.endpoint, info.dst_port))
                registration = self.decode_kv_args_table[info.peer_id]
                target_aux_item_lens.append(registration.aux_item_lens)
                target_locations.append(registration.location)
            total_kv_pages = max(
                (info.num_kv_indices for info in infos.values()),
                default=len(kv_indices),
            )

        include_aux = self.pp_group.is_last_rank
        logical_bytes = compute_logical_transfer_bytes(
            kv_args=self.kv_args,
            kv_indices=kv_indices,
            index_slice=index_slice,
            total_kv_pages=total_kv_pages,
            state_indices=state_indices,
            aux_index=aux_index,
            num_kv_tokens=num_kv_tokens,
            include_aux=include_aux,
            kv_replica_factor=self.get_kv_replica_factor(),
            target_aux_item_lens=target_aux_item_lens,
        )
        selected_link = self._select_transfer_link(
            logical_bytes=logical_bytes,
            target_locations=target_locations,
        )
        # TODO(sgl-project/sglang#49916): Model shared-link contention and ongoing traffic.
        latency_s = modeled_transfer_latency_s(
            logical_bytes=logical_bytes,
            bandwidth_gb_per_s=(
                self.bandwidth_gb_per_s
                if selected_link is None
                else selected_link.bandwidth_gb_per_s
            ),
            fixed_latency_ms=(
                self.fixed_latency_ms
                if selected_link is None
                else selected_link.fixed_latency_ms
            ),
        )
        send_record = self.send_record(bootstrap_room)
        self.note_send_enqueued(bootstrap_room)
        task = SimulatorTransferTask(
            room=bootstrap_room,
            is_last_chunk=is_last_chunk,
            targets=targets,
            target_aux_item_lens=target_aux_item_lens,
            logical_bytes=logical_bytes,
            modeled_latency_s=latency_s,
            link_name=None if selected_link is None else selected_link.name,
            aux_index=aux_index if include_aux else None,
            send_record=send_record,
        )
        self._schedule_transfer(task)
        self.update_status(bootstrap_room, KVPoll.Transferring)

    def _peer_id(self) -> str:
        return f"{self.local_ip}:{self.rank_port}"

    def _select_transfer_link(
        self,
        *,
        logical_bytes: int,
        target_locations: List[Optional[List[str]]],
    ) -> Optional[SimulatorLink]:
        if (
            self.topology is None
            or self.location is None
            or not target_locations
            or any(location is None for location in target_locations)
        ):
            return None
        links = [
            self.topology.link_between(self.location, location)
            for location in target_locations
        ]
        return max(
            links,
            key=lambda link: (
                modeled_transfer_latency_s(
                    logical_bytes=logical_bytes,
                    bandwidth_gb_per_s=link.bandwidth_gb_per_s,
                    fixed_latency_ms=link.fixed_latency_ms,
                ),
                link.name,
            ),
        )

    def _schedule_transfer(self, task: SimulatorTransferTask) -> None:
        with self._transfer_condition:
            now = time.monotonic()
            starts_at = max(now, self._room_deadlines.get(task.room, now))
            deadline = starts_at + task.modeled_latency_s
            self._room_deadlines[task.room] = deadline
            self._next_transfer_sequence += 1
            heapq.heappush(
                self._transfer_heap,
                (deadline, self._next_transfer_sequence, task),
            )
            stats = self.transfer_stats.setdefault(task.room, SimulatorTransferStats())
            stats.logical_bytes += task.logical_bytes
            stats.modeled_latency_s += task.modeled_latency_s
            stats.chunks += 1
            if self.enable_deferred_decode_kv_release:
                self._staging_outstanding[task.room] = (
                    self._staging_outstanding.get(task.room, 0) + 1
                )
            self._transfer_condition.notify()

    def _transfer_worker(self) -> None:
        while True:
            with self._transfer_condition:
                while not self._transfer_heap and not self._stopped:
                    self._transfer_condition.wait()
                if self._stopped:
                    return
                deadline, _, task = self._transfer_heap[0]
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._transfer_condition.wait(remaining)
                    continue
                heapq.heappop(self._transfer_heap)
                if self._room_deadlines.get(task.room) == deadline:
                    self._room_deadlines.pop(task.room, None)

            try:
                self._complete_transfer(task)
            except Exception as exc:
                reason = f"Simulator transfer failed: {exc}"
                logger.exception("Simulator transfer failed for room %s", task.room)
                self.record_failure(task.room, reason)
                self.update_status(task.room, KVPoll.Failed)
                try:
                    self.send_kv_status_message(
                        targets=task.targets,
                        bootstrap_room=task.room,
                        status=KVPoll.Failed,
                        failure_reason=reason,
                    )
                except Exception:
                    logger.exception(
                        "Failed to report simulator transfer failure for room %s",
                        task.room,
                    )

    def _complete_transfer(self, task: SimulatorTransferTask) -> None:
        self.note_send_finished(task.room, task.send_record)
        self._mark_transfer_quiescent(task.room)
        status = self.request_status.get(task.room)
        if task.is_last_chunk and status not in (None, KVPoll.Failed):
            self._send_aux_metadata(task)
            self.conclude_transfer(
                bootstrap_room=task.room,
                status=KVPoll.Success,
                targets=task.targets,
            )

    def _send_aux_metadata(self, task: SimulatorTransferTask) -> None:
        if task.aux_index is None:
            return
        for (endpoint, port), remote_item_lens in zip(
            task.targets, task.target_aux_item_lens
        ):
            payloads = [
                AuxDataCodec.serialize_data_from_buffer(
                    ptr + item_len * task.aux_index, item_len
                )
                for ptr, item_len in zip(
                    self.kv_args.aux_data_ptrs[: len(remote_item_lens)],
                    self.kv_args.aux_item_lens[: len(remote_item_lens)],
                )
            ]
            message = [
                _AUX_DATA_TAG,
                str(task.room).encode("ascii"),
                msgspec.msgpack.encode(payloads),
            ]
            address = NetworkAddress(endpoint, port)
            self._send_multipart_locked(
                address.to_tcp(), message, is_ipv6=address.is_ipv6
            )

    def _mark_transfer_quiescent(self, room: int) -> None:
        targets = None
        if self.enable_deferred_decode_kv_release:
            with self._transfer_condition:
                self._staging_outstanding[room] -= 1
                if self._staging_outstanding[room] <= 0:
                    self._staging_outstanding.pop(room, None)
                    if room not in self._deferred_ack_poisoned_rooms:
                        targets = self._deferred_ack_targets.pop(room, None)
        self._send_abort_acks(room, targets)

    def _handle_registration_message(self, msg: List[bytes]) -> None:
        if len(msg) != 2:
            raise ValueError("Malformed simulator registration message")
        registration = msgspec.msgpack.decode(msg[1], type=SimulatorRegistration)
        self.decode_kv_args_table[registration.peer_id] = registration

    def _handle_transfer_message(self, msg: List[bytes]) -> None:
        if len(msg) != 2:
            raise ValueError("Malformed simulator transfer message")
        info = msgspec.msgpack.decode(msg[1], type=SimulatorTransferInfo)
        if info.peer_id not in self.decode_kv_args_table:
            reason = f"Simulator decode peer {info.peer_id} was not registered"
            self.record_failure(info.room, reason)
            self.update_status(info.room, KVPoll.Failed)
            self.send_kv_status_message(
                targets=[(info.endpoint, info.dst_port)],
                bootstrap_room=info.room,
                status=KVPoll.Failed,
                failure_reason=reason,
            )
            return

        registration = self.decode_kv_args_table[info.peer_id]
        if any(
            src > dst
            for src, dst in zip(self.kv_args.aux_item_lens, registration.aux_item_lens)
        ):
            reason = f"Simulator metadata buffers are too small on decode peer {info.peer_id}"
            self.record_failure(info.room, reason)
            self.update_status(info.room, KVPoll.Failed)
            self.send_kv_status_message(
                targets=[(info.endpoint, info.dst_port)],
                bootstrap_room=info.room,
                status=KVPoll.Failed,
                failure_reason=reason,
            )
            return
        self.check_peer_aux_buffer_count(info.peer_id, len(registration.aux_item_lens))

        with self.transfer_lock:
            current = self.request_status.get(info.room)
            if current is not None and current != KVPoll.Bootstrapping:
                return
            infos = self.transfer_infos.setdefault(info.room, {})
            infos[info.peer_id] = info
            if len(infos) < info.required_dst_info_num:
                return
            self.resolve_kv_replica_factor(infos)
            self.req_to_decode_prefix_len[info.room] = info.decode_prefix_len or 0
            self.update_status(info.room, KVPoll.WaitingForInput)

    def _handle_aux_data_message(self, msg: List[bytes]) -> None:
        if len(msg) != 3:
            raise ValueError("Malformed simulator auxiliary-data message")
        room = int(msg[1].decode("ascii"))
        aux_index = self._decode_aux_indices.get(room)
        if aux_index is None:
            self.record_failure(room, "Simulator auxiliary metadata has no destination")
            self.update_status(room, KVPoll.Failed)
            return
        payloads = msgspec.msgpack.decode(msg[2], type=List[bytes])
        if len(payloads) > len(self.kv_args.aux_item_lens):
            self.record_failure(
                room,
                "Simulator auxiliary metadata has more buffers than the decode peer",
            )
            self.update_status(room, KVPoll.Failed)
            return
        for buffer_index, (payload, item_len) in enumerate(
            zip(payloads, self.kv_args.aux_item_lens)
        ):
            if len(payload) > item_len:
                self.record_failure(
                    room,
                    "Simulator auxiliary metadata item exceeds its decode buffer",
                )
                self.update_status(room, KVPoll.Failed)
                return
            AuxDataCodec.deserialize_data_to_buffer(
                self.kv_args, buffer_index, aux_index, payload
            )

    def _handle_abort_message(self, msg: List[bytes]) -> None:
        notification = AbortNotification.from_zmq(msg)
        if notification is None:
            return
        room = notification.room
        current = self.request_status.get(room)
        if current is not None and current != KVPoll.Success:
            self.update_status(room, KVPoll.Failed)

        target: Optional[AckTarget] = notification.deferred_ack_target()
        if not self.enable_deferred_decode_kv_release or target is None:
            return
        with self._transfer_condition:
            if room in self._deferred_ack_poisoned_rooms:
                return
            if self._staging_outstanding.get(room, 0) > 0:
                self.register_deferred_ack_target(room, target)
                return
        self._send_abort_ack(room, target)

    def _start_prefill_control_worker(self) -> None:
        recv = self._make_worker_recv(self.server_socket)

        def control_worker() -> None:
            while not self._stopped:
                try:
                    msg = recv()
                    if msg is None:
                        continue
                    if msg[0] == ABORT_TAG:
                        self._handle_abort_message(msg)
                    elif msg[0] == _REGISTER_TAG:
                        self._handle_registration_message(msg)
                    elif msg[0] == _TRANSFER_TAG:
                        self._handle_transfer_message(msg)
                    else:
                        logger.warning("Dropping unknown simulator control message")
                except Exception:
                    if self._stopped:
                        return
                    logger.exception("Simulator prefill control worker failed")

        worker = threading.Thread(
            target=control_worker,
            daemon=True,
            name=(f"simulator-control-dp{self.system_dp_rank}-tp{self.attn_tp_rank}"),
        )
        worker.start()
        self._worker_threads.append(worker)

    def _start_decode_status_worker(self) -> None:
        recv = self._make_worker_recv(self.server_socket)

        def status_worker() -> None:
            while not self._stopped:
                try:
                    msg = recv()
                    if msg is None:
                        continue
                    if msg[0] == _AUX_DATA_TAG:
                        self._handle_aux_data_message(msg)
                        continue
                    if self.handle_abort_ack_message(msg):
                        continue
                    parsed = self.parse_kv_status_message(msg)
                    if parsed is None:
                        logger.warning("Dropping malformed simulator status message")
                        continue
                    room, status, prefill_rank, reason = parsed
                    self.apply_prefill_status(
                        bootstrap_room=room,
                        status=status,
                        prefill_rank=prefill_rank,
                        failure_reason=reason,
                    )
                except Exception:
                    if self._stopped:
                        return
                    logger.exception("Simulator decode status worker failed")

        worker = threading.Thread(
            target=status_worker,
            daemon=True,
            name=(f"simulator-status-dp{self.system_dp_rank}-tp{self.attn_tp_rank}"),
        )
        worker.start()
        self._worker_threads.append(worker)

    def pop_transfer_stats(self, room: int) -> Optional[SimulatorTransferStats]:
        return self.transfer_stats.pop(room, None)

    def _cancel_pending_transfers(self) -> None:
        with self._transfer_condition:
            pending = [entry[2] for entry in self._transfer_heap]
            self._transfer_heap.clear()
            self._room_deadlines.clear()
            self._transfer_condition.notify_all()
        for task in pending:
            if (
                task.room in self.request_status
                and self.request_status[task.room] != KVPoll.Success
            ):
                self.update_status(task.room, KVPoll.Failed)
            self.note_send_finished(task.room, task.send_record)
            self._mark_transfer_quiescent(task.room)

    def teardown(self) -> None:
        self._stopped = True
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            self._heartbeat_shutdown.set()
        self._cancel_pending_transfers()
        for worker in self._worker_threads:
            worker.join(timeout=3)
        self._worker_threads = []
        self.server_socket.close(linger=0)
        self._zmq_ctx.destroy(linger=0)
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            CommonKVReceiver.close_all_sockets()


class SimulatorKVSender(CommonKVSender):
    def get_transfer_metric(self) -> KVTransferMetric:
        stats = self.kv_mgr.transfer_stats.get(self.bootstrap_room)
        if stats is None:
            return KVTransferMetric()
        return KVTransferMetric(
            transfer_latency_s=stats.modeled_latency_s,
            transfer_total_bytes=stats.logical_bytes,
        )

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.pop_transfer_stats(self.bootstrap_room)


class SimulatorKVReceiver(CommonKVReceiver):
    def clear(self) -> None:
        super().clear()
        self.kv_mgr._decode_aux_indices.pop(self.bootstrap_room, None)


class SimulatorKVBootstrapServer(CommonKVBootstrapServer):
    pass
