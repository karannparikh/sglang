from __future__ import annotations

import ast
import importlib
import json
import os
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import msgspec

EXECUTED = "executed"
MODELED = "modeled"
WARN_NO_OP = "warn-no-op"
REJECTED = "rejected"

VALID_CATEGORIES = frozenset({EXECUTED, MODELED, WARN_NO_OP, REJECTED})
VALID_SUBSYSTEMS = frozenset(
    {
        "attention-kernels",
        "compute-predictor",
        "cuda-runtime",
        "device-runtime",
        "disaggregation",
        "host-cache",
        "logical-memory",
        "memory-saver",
        "model-config",
        "model-loading",
        "moe-runtime",
        "parallel-runtime",
        "scheduler",
        "speculative-decoding",
    }
)

SOURCE_ARG_GROUPS = (
    ("sglang.srt.arg_groups.fields.device", "Device"),
    ("sglang.srt.arg_groups.fields.disagg", "Disagg"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecKernel"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecMamba"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecGraph"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecComm"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecMoe"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecOverlap"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecOffload"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecDeterministic"),
    ("sglang.srt.arg_groups.fields.memory", "Memory"),
    ("sglang.srt.arg_groups.fields.model", "Model"),
    ("sglang.srt.arg_groups.fields.parallel", "Parallel"),
    ("sglang.srt.arg_groups.fields.schedule", "Schedule"),
    ("sglang.srt.arg_groups.fields.spec", "Spec"),
)

NAMED_CROSS_CUTTING_FIELDS = frozenset(
    {
        "enable_fp32_lm_head",
        "enable_tf32_matmul",
        "enable_memory_saver",
        "enable_weights_cpu_backup",
        "enable_draft_weights_cpu_backup",
    }
)

PHYSICAL_SERVER_ARG_OVERRIDES = MappingProxyType(
    {
        "device": "cpu",
        "load_format": "dummy",
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "attention_backend": "torch_native",
        "prefill_attention_backend": "torch_native",
        "decode_attention_backend": "torch_native",
        "sampling_backend": "pytorch",
        "disaggregation_transfer_backend": "simulator",
    }
)

LOGICAL_TOPOLOGY_FIELDS = frozenset(
    {"tp_size", "ep_size", "dp_size", "pp_size", "attn_cp_size"}
)

_ALLOWED_REJECTED_VALUES = MappingProxyType(
    {
        "device": frozenset({"cpu"}),
        "load_format": frozenset({"dummy"}),
    }
)

_REQUESTED_VALUES_ENV = "SGLANG_SIMULATOR_REQUESTED_SERVER_ARGS"
_POLICY_OUTCOMES_ENV = "SGLANG_SIMULATOR_POLICY_OUTCOMES"


class SimulatorArgPolicy(msgspec.Struct, frozen=True):
    category: str
    subsystem: str
    rationale: str
    consumer_paths: tuple[str, ...] = ()


class SimulatorPolicyEvaluation(msgspec.Struct, frozen=True):
    warning_fields: tuple[str, ...]
    warning_messages: tuple[str, ...]
    modeled_values: dict[str, Any]


class SimulatorPolicyError(ValueError):
    pass


_registry: dict[str, SimulatorArgPolicy] = {}
_registered_field_sequence: list[str] = []


def _register(
    fields: tuple[str, ...],
    *,
    category: str,
    subsystem: str,
    rationale: str,
    consumer_paths: tuple[str, ...] = (),
) -> None:
    for field in fields:
        if field in _registry:
            raise RuntimeError(f"Duplicate simulator classification for {field!r}")
        _registered_field_sequence.append(field)
        _registry[field] = SimulatorArgPolicy(
            category=category,
            subsystem=subsystem,
            rationale=rationale,
            consumer_paths=consumer_paths,
        )


_register(
    ("tp_size", "ep_size", "dp_size", "pp_size", "attn_cp_size"),
    category=MODELED,
    subsystem="compute-predictor",
    rationale=(
        "The requested deployment width is retained as logical predictor topology "
        "while the host runtime stays single-process."
    ),
    consumer_paths=(
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/utils.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py",
    ),
)
_register(
    ("load_balance_method",),
    category=WARN_NO_OP,
    subsystem="parallel-runtime",
    rationale="The physical simulator runtime has one scheduler replica.",
)
_register(
    (
        "nccl_port",
        "dist_timeout",
        "dist_init_addr",
        "gated_launch_port",
        "nnodes",
        "node_rank",
        "dcp_size",
        "pp_max_micro_batch_size",
        "pp_async_batch_depth",
        "attn_dp_size",
        "moe_dp_size",
        "dwdp_size",
        "dcp_comm_backend",
        "dcp_replicate_q_proj",
        "enable_prefill_cp",
        "cp_strategy",
        "cp_kv_layout",
        "enable_cp_tp_group_sharing",
        "enable_dsa_cache_layer_split",
        "enable_cp_decode_attn_tp",
        "enable_kv_cache_sharding",
        "enable_dp_attention",
        "enable_dp_attention_local_control_broadcast",
        "enable_dp_lm_head",
        "enable_tp_lm_head_all_to_all",
        "enable_attn_tp_input_scattered",
        "enable_shared_experts_attn_tp",
        "shared_experts_tp_size",
        "enable_dense_mlp_attn_tp",
        "enable_layernorm_sp",
        "layernorm_sp_min_tokens",
        "enable_megamoe_for_layernorm_sp",
        "disable_attn_tp_gather",
        "enable_p2p_check",
        "moe_dense_tp_size",
        "ep_join_rank_offset",
        "elastic_ep_initial_size",
        "max_ep_size",
    ),
    category=REJECTED,
    subsystem="parallel-runtime",
    rationale=(
        "This option requires physical distributed groups or a logical model the "
        "simulator does not implement."
    ),
)

_register(
    (
        "disaggregation_transfer_backend",
        "disaggregation_simulator_bandwidth_gb_per_s",
        "disaggregation_simulator_fixed_latency_ms",
        "disaggregation_simulator_topology",
        "disaggregation_simulator_location",
    ),
    category=MODELED,
    subsystem="disaggregation",
    rationale="The simulator PD backend consumes logical bytes and link latency.",
    consumer_paths=(
        "python/sglang/srt/managers/disagg_service.py",
        "python/sglang/srt/disaggregation/simulator/conn.py",
    ),
)
_register(
    (
        "disaggregation_mode",
        "disaggregation_decode_allocation_policy",
        "disaggregation_bootstrap_port",
        "disaggregation_decode_enable_radix_cache",
        "disaggregation_decode_enable_offload_kvcache",
        "disaggregation_decode_retraction_backup",
        "num_reserved_decode_tokens",
        "disaggregation_decode_extra_slots",
        "disaggregation_decode_polling_interval",
        "enable_pd_role_switch",
        "optimistic_prefill_attempts",
    ),
    category=EXECUTED,
    subsystem="disaggregation",
    rationale="Native PD control-plane and scheduler state transitions run unchanged.",
)
_register(
    (
        "decoupled_spec_bind_endpoint",
        "decoupled_spec_connect_endpoints",
        "decoupled_spec_rank",
        "decoupled_spec_role",
        "disaggregation_enable_kv_checksum",
        "disaggregation_ib_device",
        "encoder_only",
        "language_only",
        "language_model_only",
        "encoder_transfer_backend",
        "encoder_urls",
        "encoder_bootstrap_port",
        "encoder_register_urls",
        "enable_adaptive_dispatch_to_encoder",
        "disaggregation_decode_host_receive_threshold",
    ),
    category=REJECTED,
    subsystem="disaggregation",
    rationale=(
        "This path requires real payloads, physical transport, encoder execution, "
        "or a simulator model that is not available."
    ),
)

_register(
    (
        "attention_backend",
        "decode_attention_backend",
        "enable_lean_attention",
        "enable_nvidia_megakernel",
        "enable_amd_monokernel",
        "prefill_attention_backend",
        "prefill_kv_cache_dequant_dtype",
        "sampling_backend",
        "grammar_backend",
        "fp8_gemm_runner_backend",
        "fp4_gemm_runner_backend",
        "bf16_gemm_backend",
        "dsa_prefill_backend",
        "dsv4_prefill_backend",
        "dsa_triton_union",
        "dsa_decode_backend",
        "dsv4_attn_backend",
        "dsa_paged_mqa_logits_backend",
        "dsa_topk_backend",
        "disable_flashinfer_autotune",
        "flashinfer_autotune_skip_ops",
        "triton_attention_reduce_in_fp32",
        "triton_attention_num_kv_splits",
        "triton_attention_split_tile_size",
        "flashinfer_mla_disable_ragged",
        "enable_fused_qk_norm_rope",
        "enable_precise_embedding_interpolation",
        "enable_deepseek_v4_fp4_indexer",
        "enable_fp4_indexer",
        "enable_dsa_fused_indexer",
    ),
    category=WARN_NO_OP,
    subsystem="attention-kernels",
    rationale="Model forward and sampling kernels are replaced by simulator hooks.",
)

_register(
    (
        "cuda_graph_config",
        "cuda_graph_backend_decode",
        "cuda_graph_backend_prefill",
        "cuda_graph_max_bs_decode",
        "cuda_graph_max_bs_prefill",
        "cuda_graph_max_seq_len_prefill",
        "cuda_graph_bs_decode",
        "cuda_graph_bs_prefill",
        "cuda_graph_prefill_max_context",
        "disable_prefill_cuda_graph",
        "disable_decode_cuda_graph",
        "disable_cuda_graph",
        "disable_cuda_graph_padding",
        "enable_profile_cuda_graph",
        "enable_cudagraph_gc",
        "debug_cuda_graph",
        "enable_torch_compile",
        "enable_torch_compile_debug_mode",
        "torch_compile_max_bs",
    ),
    category=WARN_NO_OP,
    subsystem="cuda-runtime",
    rationale="CUDA graph and torch.compile execution are disabled in simulator mode.",
)

_register(
    (
        "enable_layerwise_nvtx_marker",
        "disable_custom_all_reduce",
        "enforce_disable_flashinfer_allreduce_fusion",
    ),
    category=WARN_NO_OP,
    subsystem="parallel-runtime",
    rationale="No physical collective or model kernel is executed.",
)
_register(
    (
        "enable_nccl_nvls",
        "enable_symm_mem",
        "enable_mscclpp",
        "enable_torch_symm_mem",
        "enable_scattered_sconv",
        "pre_warm_nccl",
        "boundary_reduction",
        "enable_quant_communications",
        "enable_flashinfer_allreduce_fusion",
        "flashinfer_allreduce_fusion_backend",
        "enable_aiter_allreduce_fusion",
        "disable_aiter_allreduce_fusion_in_prefill",
        "disable_aiter_allreduce_fusion_in_decode",
    ),
    category=REJECTED,
    subsystem="parallel-runtime",
    rationale="The requested collective backend requires physical accelerator groups.",
)

_register(
    (
        "mamba_ssm_dtype",
        "mamba_max_states_per_path",
        "mamba_radix_cache_strategy",
        "uses_mamba_radix_cache",
        "mamba_track_interval",
        "enable_int8_mamba_checkpoint",
        "int8_mamba_ckpt_size",
    ),
    category=EXECUTED,
    subsystem="logical-memory",
    rationale="Native logical Mamba cache bookkeeping remains active.",
)
_register(
    (
        "mamba_backend",
        "mamba_cache_philox_rounds",
        "linear_attn_backend",
        "linear_attn_decode_backend",
        "linear_attn_prefill_backend",
        "linear_attn_verify_backend",
    ),
    category=WARN_NO_OP,
    subsystem="attention-kernels",
    rationale="The selected physical Mamba or linear-attention kernel does not run.",
)
_register(
    (
        "enable_mamba_cache_stochastic_rounding",
        "enable_linear_replayssm",
        "linear_replayssm_cache_len",
        "enable_linear_replayssm_spec",
    ),
    category=REJECTED,
    subsystem="attention-kernels",
    rationale="This feature requires physical recurrent-state kernel execution.",
)

_register(
    (
        "enable_fused_moe_sum_all_reduce",
        "moe_a2a_backend",
        "enable_w4a4_mxfp4_megamoe",
        "deepep_v2_mode",
        "moe_runner_backend",
        "flashinfer_mxfp4_moe_precision",
        "deepep_mode",
        "fuseep_mode",
        "deepep_dispatcher_output_dtype",
        "flashinfer_a2a_dispatch_type",
        "ep_num_redundant_experts",
        "ep_dispatch_algorithm",
        "init_expert_location",
        "enable_eplb",
        "eplb_algorithm",
        "eplb_rebalance_num_iterations",
        "eplb_rebalance_layers_per_chunk",
        "eplb_min_rebalancing_utilization_threshold",
        "expert_distribution_recorder_mode",
        "expert_distribution_recorder_buffer_size",
        "expert_balancedness_report_mode",
        "deepep_config",
        "moe_ascend_ops_config",
        "elastic_ep_backend",
        "enable_elastic_expert_backup",
        "mooncake_ib_device",
        "enable_waterfill",
        "ep_join_mode",
        "elastic_ep_scale_timeout",
        "disable_flashinfer_cutlass_moe_fp4_allgather",
        "disable_shared_experts_fusion",
        "enforce_shared_experts_fusion",
        "enable_paged_experts",
        "paged_experts_num_resident",
        "paged_experts_enable_disk",
        "paged_experts_host_memory_gb",
        "kt_weight_path",
        "kt_method",
        "kt_cpuinfer",
        "kt_threadpool_count",
        "kt_num_gpu_experts",
        "kt_max_deferred_experts_per_token",
    ),
    category=REJECTED,
    subsystem="moe-runtime",
    rationale="MoE kernels, expert placement, paging, and physical A2A are not simulated.",
)

_register(
    (
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "tbo_token_distribution_threshold",
    ),
    category=WARN_NO_OP,
    subsystem="scheduler",
    rationale="The simulator runs the native non-overlap scheduler event loop.",
)
_register(
    (
        "cpu_offload_gb",
        "offload_group_size",
        "offload_num_in_group",
        "offload_prefetch_step",
        "offload_mode",
        "ple_offload_embedding",
        "ple_offload_backend",
        "ple_offload_dir",
        "engram_config",
    ),
    category=REJECTED,
    subsystem="model-loading",
    rationale="Weights are never materialized, so physical weight offload is unavailable.",
)
_register(
    ("enable_deterministic_inference", "rl_on_policy_target"),
    category=WARN_NO_OP,
    subsystem="attention-kernels",
    rationale=(
        "Simulator token generation is deterministic but does not execute the "
        "requested batch-invariant physical kernels."
    ),
)

_register(
    (
        "radix_eviction_policy",
        "radix_eviction_policy_config",
        "disable_radix_cache",
        "enable_page_major_kv_layout",
        "enable_unified_memory",
        "enable_session_radix_cache",
        "enable_hierarchical_cache",
        "hicache_host_memory_mode",
        "hicache_size",
        "hicache_host_memory_fraction",
        "hicache_write_policy",
        "hicache_io_backend",
        "hicache_mem_layout",
        "hicache_storage_prefetch_policy",
        "hicache_storage_prefetch_retry_poll_interval",
        "hicache_storage_prefetch_retry_max_attempts",
        "hicache_storage_key_scheme",
        "hicache_storage_head_partition",
        "hicache_storage_layer_partition",
    ),
    category=EXECUTED,
    subsystem="host-cache",
    rationale="Native radix and HiCache control-plane bookkeeping remains active.",
)
_register(
    ("hicache_ratio", "hicache_storage_backend"),
    category=MODELED,
    subsystem="host-cache",
    rationale="The logical host/storage tiers retain capacity and key-only behavior.",
    consumer_paths=(
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/mem_pool_host.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/hicache_storage.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/manager/config.py",
        "python/sglang/srt/managers/scheduler.py",
    ),
)
_register(
    (
        "hicache_storage_backend_extra_config",
        "unified_cache_external_linker_backend",
        "unified_cache_external_linker_config",
        "hisparse_config",
        "lmcache_config_file",
        "flexkv_config_file",
    ),
    category=WARN_NO_OP,
    subsystem="host-cache",
    rationale="The associated physical backend is replaced or disabled in simulation.",
)
_register(
    (
        "radix_cache_backend",
        "enable_unified_cache_external_linker",
        "enable_linker_mla_dedup",
        "enable_hisparse",
        "enable_lmcache",
        "enable_flexkv",
    ),
    category=REJECTED,
    subsystem="host-cache",
    rationale="This cache implementation requires an unmodeled physical backend.",
)

_register(
    (
        "mem_fraction_static",
        "max_total_tokens",
        "page_size",
        "swa_full_tokens_ratio",
        "max_mamba_cache_size",
        "mamba_full_memory_ratio",
    ),
    category=MODELED,
    subsystem="logical-memory",
    rationale="These values size logical KV/state pools whose payloads use meta storage.",
    consumer_paths=(
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/model_runner.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/mem_pool_device.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/mem_pool_host.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/utils.py",
        "python/sglang/srt/mem_cache/kv_cache_configurator.py",
    ),
)
_register(
    ("disable_overlap_schedule",),
    category=WARN_NO_OP,
    subsystem="scheduler",
    rationale="The simulator always selects the native non-overlap event loop.",
)
_register(
    (
        "max_running_requests",
        "max_queued_requests",
        "chunked_prefill_size",
        "prefill_decode_interval",
        "prefill_decode_interval_min_bs",
        "enable_dynamic_chunking",
        "max_prefill_tokens",
        "prefill_max_requests",
        "schedule_policy",
        "enable_priority_scheduling",
        "disable_priority_preemption",
        "default_priority_value",
        "abort_on_priority_when_disabled",
        "schedule_low_priority_values_first",
        "priority_scheduling_preemption_threshold",
        "retraction_policy",
        "schedule_conservativeness",
        "c128_page_size",
        "swa_prefix_tails",
        "disable_hybrid_swa_memory",
        "prefill_only_disable_kv_cache",
        "disable_chunked_prefix_cache",
        "num_continuous_decode_steps",
        "scheduler_recv_interval",
        "enable_mixed_chunk",
        "enable_prefill_delayer",
        "prefill_delayer_max_delay_passes",
        "prefill_delayer_token_usage_low_watermark",
        "prefill_delayer_forward_passes_buckets",
        "prefill_delayer_wait_seconds_buckets",
        "prefill_delayer_queue_min_ratio",
        "prefill_delayer_max_delay_ms",
        "min_free_slots_delay",
    ),
    category=EXECUTED,
    subsystem="scheduler",
    rationale="Native scheduler admission, batching, priority, and timeout semantics run.",
)

_register(
    (
        "model_path",
        "context_length",
        "revision",
        "model_impl",
        "model_config_parser",
        "json_model_override_args",
        "dtype",
        "kv_cache_dtype",
    ),
    category=MODELED,
    subsystem="model-config",
    rationale="Model metadata feeds predictor geometry, logical cache sizing, or token identity.",
    consumer_paths=(
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/model_runner.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/utils.py",
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/deterministic_sampling.py",
        "python/sglang/srt/configs/model_config.py",
    ),
)
_register(
    ("trust_remote_code", "engine_info_bootstrap_port"),
    category=EXECUTED,
    subsystem="model-config",
    rationale="Configuration discovery and bootstrap control-plane behavior remains native.",
)
_register(
    ("quantization", "quantization_param_path", "qsa_indexer_dtype"),
    category=WARN_NO_OP,
    subsystem="model-config",
    rationale="No quantized weights or physical indexer cache are materialized.",
)
_register(
    (
        "load_format",
        "model_loader_extra_config",
        "is_embedding",
        "disable_normalize_embedding",
        "modelopt_quant",
        "modelopt_checkpoint_restore_path",
        "modelopt_checkpoint_save_path",
        "modelopt_export_path",
        "quantize_and_serve",
        "rl_quant_profile",
        "startup_weight_load_mode",
        "custom_weight_loader",
        "weight_loader_disable_mmap",
        "weight_loader_prefetch_checkpoints",
        "weight_loader_prefetch_num_threads",
        "weight_loader_drop_cache_after_load",
        "remote_instance_weight_loader_seed_instance_ip",
        "remote_instance_weight_loader_seed_instance_service_port",
        "remote_instance_weight_loader_send_weights_group_ports",
        "remote_instance_weight_loader_backend",
        "remote_instance_weight_loader_start_seed_via_transfer_engine",
        "modelexpress_config",
        "download_dir",
        "model_checksum",
        "delete_ckpt_after_loading",
        "decrypted_config_file",
        "decrypted_draft_config_file",
        "checkpoint_engine_wait_weights_before_ready",
        "weight_cache_mode",
        "weight_cache_socket",
        "weight_cache_timeout",
    ),
    category=REJECTED,
    subsystem="model-loading",
    rationale="Simulator startup loads configuration only and never opens model weights.",
)

_register(
    (
        "watchdog_timeout",
        "soft_watchdog_timeout",
        "sleep_on_idle",
        "custom_sigquit_handler",
        "gc_threshold",
    ),
    category=EXECUTED,
    subsystem="device-runtime",
    rationale="These host control-plane settings retain their native behavior.",
)
_register(
    ("random_seed",),
    category=WARN_NO_OP,
    subsystem="device-runtime",
    rationale="Deterministic simulator tokens use request sampling state, not the process seed.",
)
_register(
    (
        "device",
        "base_gpu_id",
        "gpu_id_step",
        "mlx_enable_sampling",
        "use_ray",
        "numa_node",
    ),
    category=REJECTED,
    subsystem="device-runtime",
    rationale="Simulator workers run as local CPU processes without accelerator placement.",
)

_register(
    tuple(sorted(NAMED_CROSS_CUTTING_FIELDS)),
    category=WARN_NO_OP,
    subsystem="memory-saver",
    rationale="No dense logits or model weights exist for these physical precision/storage controls.",
)

_register(
    (
        "speculative_algorithm",
        "uno_lora_path",
        "speculative_draft_model_path",
        "speculative_boundary_reduction",
        "speculative_draft_model_revision",
        "speculative_draft_load_format",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_dflash_block_size",
        "speculative_domino_candidate_pool_size",
        "speculative_dspark_block_size",
        "speculative_dspark_sps_table_path",
        "speculative_dspark_confidence_sts_path",
        "speculative_dspark_align_verify_tokens_to_graph_tier",
        "speculative_accept_threshold_single",
        "speculative_accept_threshold_acc",
        "speculative_use_rejection_sampling",
        "speculative_use_block_verification",
        "speculative_token_map",
        "speculative_attention_mode",
        "speculative_draft_attention_backend",
        "speculative_dsa_topk_backend",
        "speculative_draft_kv_cache_dtype",
        "speculative_dspark_lm_head_dtype",
        "speculative_draft_window_size",
        "speculative_dflash_draft_kv_ring",
        "speculative_draft_sink_size",
        "speculative_moe_runner_backend",
        "speculative_moe_a2a_backend",
        "speculative_enable_w4a4_mxfp4_megamoe",
        "speculative_draft_model_quantization",
        "enable_multi_layer_eagle",
        "speculative_adaptive",
        "speculative_adaptive_config",
        "spec_trace_dir",
        "speculative_ngram_min_bfs_breadth",
        "speculative_ngram_max_bfs_breadth",
        "speculative_ngram_match_type",
        "speculative_ngram_max_trie_depth",
        "speculative_ngram_capacity",
        "speculative_ngram_external_corpus_path",
        "speculative_ngram_external_sam_budget",
        "speculative_ngram_external_corpus_max_tokens",
    ),
    category=REJECTED,
    subsystem="speculative-decoding",
    rationale="Draft/verify compute and acceptance are not represented by the predictor.",
)

SIMULATOR_ARG_POLICY = MappingProxyType(_registry)


def _server_arg_defaults() -> dict[str, Any]:
    from sglang.srt.arg_groups.arg_utils import record_fields
    from sglang.srt.server_args import ServerArgs

    defaults = {}
    for field in record_fields(ServerArgs):
        if field.default is not msgspec.NODEFAULT:
            defaults[field.name] = field.default
        elif field.default_factory is not msgspec.NODEFAULT:
            defaults[field.name] = field.default_factory()
    return defaults


def server_arg_values(target) -> dict[str, Any]:
    if isinstance(target, Mapping):
        return {name: target[name] for name in SIMULATOR_ARG_POLICY if name in target}
    if isinstance(target, msgspec.Struct):
        return {name: getattr(target, name) for name in SIMULATOR_ARG_POLICY}
    values = vars(target)
    return {name: values[name] for name in SIMULATOR_ARG_POLICY if name in values}


def evaluate_simulator_policy(
    values: Mapping[str, Any],
    *,
    explicit_fields: set[str] | frozenset[str] = frozenset(),
    ignored_fields: set[str] | frozenset[str] = frozenset(),
    defaults: Mapping[str, Any] | None = None,
) -> SimulatorPolicyEvaluation:
    defaults = _server_arg_defaults() if defaults is None else defaults
    warning_fields = []
    warning_messages = []
    modeled_values = {}

    for name in sorted(values):
        policy = SIMULATOR_ARG_POLICY.get(name)
        if policy is None or name in ignored_fields:
            continue
        value = values[name]
        changed = name in explicit_fields or (
            name in defaults and value != defaults[name]
        )
        if policy.category == MODELED:
            modeled_values[name] = value
            continue
        if not changed:
            continue
        option = f"--{name.replace('_', '-')}"
        if policy.category == WARN_NO_OP:
            warning_fields.append(name)
            warning_messages.append(
                f"{option} is accepted but has no simulator effect: {policy.rationale}"
            )
        elif policy.category == REJECTED:
            allowed = _ALLOWED_REJECTED_VALUES.get(name, frozenset())
            if value not in allowed:
                raise SimulatorPolicyError(
                    f"{option}={value!r} is unsupported in simulator mode: "
                    f"{policy.rationale}"
                )

    return SimulatorPolicyEvaluation(
        warning_fields=tuple(warning_fields),
        warning_messages=tuple(warning_messages),
        modeled_values=modeled_values,
    )


def requested_logical_topology(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: values[name] for name in sorted(LOGICAL_TOPOLOGY_FIELDS) if name in values
    }


def record_policy_context(
    *,
    requested_values: Mapping[str, Any],
    evaluation: SimulatorPolicyEvaluation,
) -> None:
    os.environ[_REQUESTED_VALUES_ENV] = json.dumps(
        requested_logical_topology(requested_values),
        sort_keys=True,
        separators=(",", ":"),
    )
    os.environ[_POLICY_OUTCOMES_ENV] = json.dumps(
        {"warn_no_op": list(evaluation.warning_fields)},
        sort_keys=True,
        separators=(",", ":"),
    )


def requested_server_arg_values() -> dict[str, Any]:
    text = os.getenv(_REQUESTED_VALUES_ENV)
    return {} if text is None else json.loads(text)


def recorded_policy_outcomes() -> dict[str, list[str]]:
    text = os.getenv(_POLICY_OUTCOMES_ENV)
    return {"warn_no_op": []} if text is None else json.loads(text)


def policy_context_recorded() -> bool:
    return _POLICY_OUTCOMES_ENV in os.environ


def category_counts() -> dict[str, int]:
    counts = Counter(policy.category for policy in SIMULATOR_ARG_POLICY.values())
    return {category: counts.get(category, 0) for category in sorted(VALID_CATEGORIES)}


def hardware_sensitive_server_arg_fields(
    repo_root: Path | None = None,
) -> frozenset[str]:
    fields = set(NAMED_CROSS_CUTTING_FIELDS)
    for module_name, class_name in SOURCE_ARG_GROUPS:
        if repo_root is None:
            source = getattr(importlib.import_module(module_name), class_name)
            fields.update(
                field.name
                for field in msgspec.structs.fields(source)
                if not field.name.startswith("_")
            )
            continue

        source_path = repo_root / "python" / Path(module_name.replace(".", "/") + ".py")
        source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in source_tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        fields.update(
            node.target.id
            for node in class_node.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        )
    return frozenset(fields)


def registry_validation_errors(repo_root: Path | None = None) -> list[str]:
    errors = []
    current_fields = hardware_sensitive_server_arg_fields(repo_root)
    registered_fields = frozenset(SIMULATOR_ARG_POLICY)
    missing = sorted(current_fields - registered_fields)
    stale = sorted(registered_fields - current_fields)
    if missing:
        errors.append(f"Missing simulator classifications: {missing}")
    if stale:
        errors.append(f"Stale simulator classifications: {stale}")
    if len(_registered_field_sequence) != len(set(_registered_field_sequence)):
        errors.append("Duplicate simulator classifications are present")

    for name, policy in SIMULATOR_ARG_POLICY.items():
        if policy.category not in VALID_CATEGORIES:
            errors.append(f"{name}: invalid category {policy.category!r}")
        if policy.subsystem not in VALID_SUBSYSTEMS:
            errors.append(f"{name}: invalid subsystem {policy.subsystem!r}")
        if policy.category == MODELED and not policy.consumer_paths:
            errors.append(f"{name}: modeled field has no consumer assertion")
        if repo_root is not None and policy.category == MODELED:
            consumers = []
            for relative_path in policy.consumer_paths:
                path = repo_root / relative_path
                if not path.is_file():
                    errors.append(f"{name}: consumer path is missing: {relative_path}")
                    continue
                consumers.append(path.read_text(encoding="utf-8"))
            if consumers and not any(name in content for content in consumers):
                errors.append(
                    f"{name}: consumer assertion did not find the field in "
                    f"{list(policy.consumer_paths)}"
                )
    return errors


def build_startup_summary(
    *,
    predictor_name: str,
    predictor_mode: str,
    simulation_mode: str,
    scheduler_config,
    max_total_tokens: int,
    page_size: int,
    hicache_ratio: float | None,
    disaggregation_backend: str,
    disaggregation_bandwidth_gb_per_s: float,
    disaggregation_fixed_latency_ms: float,
    disaggregation_topology: str | None,
    disaggregation_location: str | None,
    deterministic_sampling_version: str,
) -> str:
    if disaggregation_topology is None:
        pd_model = {
            "bandwidth_gb_per_s": disaggregation_bandwidth_gb_per_s,
            "fixed_latency_ms": disaggregation_fixed_latency_ms,
            "type": "global-link",
        }
    else:
        topology = json.loads(disaggregation_topology)
        pd_model = {
            "levels": topology["levels"],
            "location": disaggregation_location,
            "type": "topology",
        }
    summary = {
        "classification_counts": category_counts(),
        "compute": {
            "predictor": predictor_name,
            "predictor_mode": predictor_mode,
            "simulation_mode": simulation_mode,
        },
        "deterministic_sampling": deterministic_sampling_version,
        "explicit_outcomes": recorded_policy_outcomes(),
        "logical_kv": {
            "device_payload": "meta",
            "hicache_ratio": hicache_ratio,
            "host_payload": "meta",
            "max_total_tokens": max_total_tokens,
            "page_size": page_size,
        },
        "logical_topology": {
            "attn_cp_size": scheduler_config.cp_size,
            "dp_size": scheduler_config.dp_size,
            "ep_size": scheduler_config.ep_size,
            "pp_size": scheduler_config.pp_size,
            "tp_size": scheduler_config.tp_size,
        },
        "pd": {
            "backend": disaggregation_backend,
            "model": pd_model,
        },
        "physical_runtime": dict(PHYSICAL_SERVER_ARG_OVERRIDES),
    }
    return "SGLang Simulator effective configuration: " + json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
    )
