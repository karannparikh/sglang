# SGLang Simulator

SGLang Simulator reuses SGLang's scheduler and cache implementation while
replacing model forward execution with a latency predictor. It supports
timestamped trace replay, synthetic workloads, hierarchical cache simulation,
and serving-compatible metrics without loading model weights.

See the [SGLang Simulator advanced-feature guide](../../docs/docs/advanced_features/sglang_simulator.mdx)
for the user-facing setup and serving workflow.

## Compatibility

SGLang Simulator tracks the current SGLang `main` branch and maintains compatibility
with recent SGLang releases. The current integration is validated with `v0.5.16`,
`v0.5.17`, `v0.5.18`, and `main`. Compatibility code uses API and capability
checks instead of branching on version numbers.

## Requirements

- A compatible SGLang checkout. The simulator uses the SGLang source from the
  same monorepo checkout.
- A local model directory containing model configuration files. Tokenizer files
  are also required unless tokenizer initialization is disabled.
- Predictor data for AISimulate, ML, or replay mode.

Use an official SGLang image matching the checkout when validating GPU and
runtime compatibility.

## Installation

From the SGLang repository:

```bash
pip install -e tools/sglang-simulator
```

The simulator does not install or pin a second `sglang` package. Run it from a
checkout whose `python/sglang` package is available on `PYTHONPATH`, or from a
matching official SGLang image.

### Estimator SDK

Install the AISimulate estimator SDK in a compatible environment:

```bash
pip install -e "tools/sglang-simulator[aisimulate]"
```

The extra pins `aisimulate==0.13.0`. The official wheels require Python
3.11–3.13 and cover Linux x86_64/aarch64 and macOS arm64. This release exposes
the modern `aisimulate.sdk` namespace.

AISimulate 0.13.0 requires NumPy>=2.1,<3, replacing the NumPy 1.26.4 pin
used by standalone AIConfigurator 0.10.0. Use an environment compatible with
both the SDK and your SGLang runtime. If your existing runtime requires
NumPy 1.x, prepare a separate compatible environment.

For a new environment, install the SGLang runtime and its dependencies from
the same checkout before installing the estimator extra above. The extra
installs the simulator and estimator SDK, not the SGLang runtime dependencies;
adding source directories to `PYTHONPATH` does not install those dependencies.
No old-package uninstall is needed in a fresh environment.

Use the `aisimulate` predictor name for the existing static op-level path,
including phase scale factors.
The shared memory-estimation path also uses this SDK for ML/replay predictors.

#### Migrating from AIConfigurator

To upgrade an existing compatible environment, activate that environment and
remove the standalone distributions before installing the pinned SDK extra,
as required by the AISimulate 0.13.0 migration instructions:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install -e "tools/sglang-simulator[aisimulate]"
```

Replace each AIConfigurator artifact with its AISimulate counterpart:

| Old (AIConfigurator) | New (AISimulate) |
| --- | --- |
| Install extra `[aic]` | Install extra `[aisimulate]` |
| Predictor name `"aiconfigurator"` | Predictor name `"aisimulate"` |
| `sglang_simulator.time_predictor.aiconfigurator` | `sglang_simulator.time_predictor.aisimulate` |
| `AIConfiguratorTimePredictor` | `AISimulateTimePredictor` |
| `examples/sim_configs/aic_sol.json` | `examples/sim_configs/aisimulate_sol.json` |
| `examples/sim_configs/aic_silicon.json` | `examples/sim_configs/aisimulate_silicon.json` |

There is no compatibility layer: a simulator configuration that still uses the
`"aiconfigurator"` predictor name fails with an explicit error naming the
replacement. Update the configuration instead of relying on a fallback.

The migration keeps the SGLang-side parameter construction, uneven-batch
compensation, and phase scale factors. It does not promise unchanged numerical
results between AIConfigurator and AISimulate performance databases; re-baseline
against your own measurements.

The supplied SOL/SILICON examples select backend data version `0.5.10`; this
is a performance-database slot, not the version of the SGLang checkout running
the simulator.

## Quick start

The maintained tests define the supported first-version scope:

- [`test/test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py):
  direct Python use of the repository-level
  [`SGLangBenchmarkRunner`](../../benchmark/simulator/bench_runner.py);
- [`test/test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py):
  server plus benchmark-client use through the HTTP serving path with the
  installed estimator SDK, ML, and replay predictors and ShareGPT or
  timestamped traffic;
- [`test/test_simulation_offline_blocking.py`](test/test_simulation_offline_blocking.py):
  equivalent logical results in `OFFLINE` and `BLOCKING` modes;
- [`test/test_simulation_cache_hit_ratio.py`](test/test_simulation_cache_hit_ratio.py):
  reusable-prefix accounting and cache-tier hit metrics across repeated runs;
- [`test/test_aisimulate_predictor.py`](test/test_aisimulate_predictor.py):
  real-SDK prefill/decode prediction and prefix reuse in SOL and SILICON modes.

From `tools/sglang-simulator`:

```bash
python3 -m pytest -q test/test_simulation_sglang_runner.py
python3 -m pytest -q test/test_simulation_sglang_serving.py
```

Read these tests as the minimal maintained examples for constructing a dataset,
running a benchmark, starting a simulator server, sending programmatic, ShareGPT,
or timestamped traffic, comparing execution modes, and collecting request,
latency, throughput, and prefix-cache metrics.

The serving and offline-blocking tests use the AISimulate example
configurations `aisimulate_sol` and `aisimulate_silicon`, which pin the
`0.5.10` backend slot that the AISimulate performance databases publish. With
no estimator SDK installed, the SDK-dependent cases fail at server startup;
it is not skipped. The CPU CI workflow runs the full suite in a single
AISimulate environment with the real SDK installed.

## Serving mode

Choose a fresh output directory and export it in the server terminal before
starting the server:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
test ! -e "$SIMULATOR_OUTPUT_DIR"
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /absolute/path/to/model \
  --sim-config-path /absolute/path/to/simulator.json \
  --port 30000
```

In the benchmark terminal, export the same output directory before sending
timestamped traffic with the simulator-aware benchmark adapter:

```bash
cd /path/to/sglang
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 benchmark/simulator/bench_serving.py \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /absolute/path/to/model \
  --dataset-name autobench \
  --dataset-path /absolute/path/to/trace.jsonl \
  --use-trace-timestamps \
  --num-prompts 100 \
  --profile \
  --output-file "$SIMULATOR_OUTPUT_DIR/benchmark.json"
```

The server and benchmark are separate processes, so exporting
`SGLANG_SIMULATOR_OUTPUT_DIR` in the server terminal does not configure the
benchmark terminal. The benchmark adapter reads `metrics.json` from this path
after profiling and uses those server-side logical-time metrics for its serving
table and output file. If the benchmark points at another directory, it may show
unrelated stale metrics or client wall-clock values. Use the same fresh path in
both terminals for every run.

The simulator always runs the physical SGLang runtime with
`tp_size=ep_size=dp_size=pp_size=1` and both attention/decode
context-parallel sizes set to `1`. Requested `--tp-size`, `--ep-size`,
`--dp-size`, `--pp-size`, and `--attn-cp-size` values are preserved as the
logical deployment topology used by the predictor. Values in the simulator
JSON `scheduler` section take precedence over those CLI values. Decode-only
`--dcp-size` is rejected when it is not `1` because no predictor model consumes
it; the simulator never silently collapses an unmodeled parallelism request.

Other server options are normal SGLang command-line arguments. For direct Python
integration, see
[`test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py); for the
process/HTTP path, see
[`test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py).

### Hardware option policy

Every hardware-sensitive field in the maintained ServerArgs inventory has one
explicit simulator classification:

| Classification | Behavior |
|---|---|
| `executed` | Native scheduler or control-plane semantics run normally. |
| `modeled` | A named simulator subsystem consumes the value, such as the compute predictor, logical KV pools, deterministic sampling, PD transport, or topology model. |
| `warn-no-op` | A non-default or explicitly supplied value is accepted with one field-specific startup warning because no meaningful simulator behavior exists yet. Untouched defaults do not warn. |
| `rejected` | A non-default request fails before worker startup with the argument name and reason because accepting it would imply unavailable physical execution. |

The registry is in
[`hardware_policy.py`](src/sglang_simulator/hardware_policy.py). Its deterministic
inclusion rule is the complete `Device`, `Disagg`, `ExecKernel`, `ExecMamba`,
`ExecGraph`, `ExecComm`, `ExecMoe`, `ExecOverlap`, `ExecOffload`,
`ExecDeterministic`, `Memory`, `Model`, `Parallel`, `Schedule`, and `Spec`
ServerArgs field groups, plus the named precision and memory-saver fields from
`ExecFeatures`. A drift test fails for missing or stale entries, duplicate or
invalid classifications, and modeled fields without a source-level consumer
assertion. When adding a field to one of these groups, add exactly one registry
entry with a concise rationale, subsystem, and implementation consumer when
the category is `modeled`.

Simulator activation classifies arguments before ServerArgs resolution and
forces only the physical single-process substrate (`cpu`, dummy loading,
torch-native attention, PyTorch sampling, no CUDA graph, and simulator PD
transport). This applies to the standalone launcher, direct benchmark runner,
and early-startup external harness bridge.

Startup emits one stable JSON summary prefixed with
`SGLang Simulator effective configuration:`. It reports the predictor and
mode, deterministic-token version, logical KV capacity and page geometry,
logical topology, selected PD link or topology model, classification counts,
explicit warn-no-op outcomes, and forced physical runtime. It intentionally
does not dump all ServerArgs or secret-bearing values.

### KV memory model

The simulator keeps SGLang's native token/page allocators, request-to-token
mapping, radix cache, cache keys, references, eviction, and events. Native MHA,
MLA, packed single-pool, and host K/V payload tensors retain their logical
shapes but use metadata-only storage, so modeled cache bytes are not physically
allocated. Increasing `max_total_tokens` still increases allocator page
bookkeeping; it does not allocate the corresponding K/V values.

Aggregate `metrics.json` output also includes bounded simulator observability:
`simulator_predicted_compute_time_s`, `simulator_logical_transfer_bytes`,
`simulator_logical_transfer_time_s`,
`simulator_virtual_kv_capacity_tokens`, and
`simulator_virtual_kv_page_size`. Native scheduler, KV-cache, and FPM metrics
remain unchanged.

### PD transfer model

Simulator activation selects SGLang's `simulator` disaggregation transfer
backend. It keeps the common bootstrap registry, sender/receiver lifecycle,
chunking, metadata exchange, polling states, aborts, timeouts, cleanup, and
transfer metrics, but does not register memory or copy payload values.

Each transfer chunk is modeled independently as:

```text
fixed latency + logical bytes / bandwidth
```

Logical bytes come from the real KV and state item sizes and the actual indices
sent by SGLang. The backend stores only those counts and small control messages.
Configure the initial single-link model with
`--disaggregation-simulator-bandwidth-gb-per-s` and
`--disaggregation-simulator-fixed-latency-ms`.

Workers can optionally advertise
`--disaggregation-simulator-location region/rack/node` and share a topology:

```json
{
  "levels": ["region", "rack", "node"],
  "links": {
    "region": {"bandwidth_gb_per_s": 10, "fixed_latency_ms": 0.5},
    "rack": {"bandwidth_gb_per_s": 50, "fixed_latency_ms": 0.05},
    "node": {"bandwidth_gb_per_s": 400, "fixed_latency_ms": 0.01},
    "default": {"bandwidth_gb_per_s": 1, "fixed_latency_ms": 5}
  }
}
```

Pass the compact JSON document through
`--disaggregation-simulator-topology`. The deepest shared location level selects
the link; endpoints with no shared level use `default`. If the topology or
either endpoint location is absent, transfers keep the global single-link
settings. The harness remains responsible for launching and placing workers.
The simulator does not model shared-link contention, routing, failures,
retransmission, or response-KV return.

### H200 lab reference profiles

The opt-in `h200-lab-disaggregated-v1` profile reproduces the capacity and
sampled engine-latency inputs from one internal 8-GPU H200 lab setup. It is a
reference profile, not a claim about every H200 deployment.

| Setting | Profile value |
| --- | --- |
| Logical TP width | 8 |
| KV page size | 256 tokens |
| KV capacity | 15,509 pages / 3,970,304 tokens |
| Context length | 1,048,576 |
| Maximum running requests | 32 |
| Maximum/chunked prefill | 8,192 tokens |
| KV dtype | FP8 E4M3 |
| Default timing seed | 0 |

The profile owns the `h200-lab-sampled-v1` predictor. Prefill samples are
conditioned on the actual recomputed tokens in each native scheduler step;
decode samples are conditioned on the active decode batch size. Workload
quantiles are interpolated between measured anchors and clamp above the final
anchor. Sub-8K prefill work is prorated around a fixed 4 ms overhead, while a
zero-token cache hit costs zero compute time.

Use one of the ready simulator configs:

```text
tools/sglang-simulator/examples/sim_configs/h200_lab_two_zones.json
tools/sglang-simulator/examples/sim_configs/h200_lab_mixed.json
```

They select the same capacity/timing profile and one of two named link presets:

| Preset | Levels and links |
| --- | --- |
| `h200-lab-two-zones-v1` | node 450 GB/s + 0.01 ms; rack 50 GB/s + 0.05 ms; zone 12.5 GB/s + 0.5 ms; default/inter-zone 1.25 GB/s + 5 ms |
| `h200-lab-mixed-v1` | node 450 GB/s + 0.01 ms; rack 50 GB/s + 0.05 ms; default 12.5 GB/s + 0.5 ms |

A Dynamo harness selects the config in each worker environment and still owns
worker launch and placement:

```yaml
env:
  SGLANG_SIMULATOR_BOOTSTRAP: "1"
  SGLANG_SIMULATOR_CONFIG_PATH: /opt/sglang/tools/sglang-simulator/examples/sim_configs/h200_lab_two_zones.json
  SGLANG_SIMULATOR_LOGICAL_RANK: "0"
args:
  - --disaggregation-mode
  - prefill
  - --disaggregation-simulator-location
  - zone-a/rack-1/node-1
```

Use a distinct stable non-negative `SGLANG_SIMULATOR_LOGICAL_RANK` for every
logical worker. The profile combines it with `--random-seed` in a process-local
random stream, so reruns are reproducible and ranks do not draw the same
sequence. Decode workers use the same config with
`--disaggregation-mode decode` and their own location and logical rank. For the
mixed preset, locations have two components such as `rack-1/node-2`.

Profile values are defaults. Explicit CLI/runtime ServerArgs values override
them. `profile_overrides` in the simulator JSON handles external harnesses that
must explicitly restore a value equal to the ordinary SGLang default; a
different CLI/runtime value for the same field is rejected as ambiguous.
Existing simulator JSON `scheduler` topology values retain their higher
predictor precedence. A profile cannot be combined with a separate `predictor`
section, and a topology preset cannot be combined with an explicit inline
topology.

The sampled data applies only to disaggregated `prefill` and `decode` workers.
Selecting this profile for an aggregated/non-PD worker fails at startup rather
than presenting the measurements as aggregated-worker timing.

## Simulation modes

| Mode | Behavior |
|---|---|
| `OFFLINE` | Advances the simulator's logical clock without sleeping. |
| `BLOCKING` | Sleeps for predicted forward and visible L2-to-L1 load latency. |

Use server-side simulator metrics for comparisons. Client wall-clock duration is
not the simulated timeline in `OFFLINE` mode. When using the benchmark adapter,
make sure its `SGLANG_SIMULATOR_OUTPUT_DIR` matches the server's output directory
so the printed table and `benchmark.json` are sourced from the current run's
`metrics.json`.

## Configuration

A simulator configuration has three sections:

```json
{
  "platform": {
    "accelerator": {"name": "h20_sxm"},
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 1
  },
  "predictor": {
    "name": "replay",
    "database_path": "/absolute/path/to/replay_table.json"
  },
  "scheduler": {
    "tp_size": 4,
    "ep_size": 4,
    "dp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "cp_style": "none",
    "data_type": "BF16",
    "kv_cache_data_type": "BF16",
    "backend_name": "sglang"
  }
}
```

- `platform` describes the simulated accelerator and storage bandwidth.
- `predictor` selects forward-latency prediction.
- `scheduler` describes the real target deployment topology and backend metadata.
  `tp_size`, `ep_size`, `dp_size`, `pp_size`, and `cp_size` are modeled values;
  they do not launch physical workers. For the AISimulate predictor, `tp_size`
  is converted to attention TP after removing modeled DP and CP, while
  `cp_size` is passed as AISimulate context parallelism. `cp_style` uses the
  AISimulate values such as `none`, `allgather`, `ulysses`, or `ring`.
  Decode-only `dcp_size` has no separate AISimulate field and is not modeled
  yet.

### Prefix-cache accuracy

Prefix-cache hit accuracy is highly sensitive to `max_total_tokens`. It controls
the simulated device KV-cache capacity and participates in hierarchical host-cache
sizing, so a mismatch changes eviction timing and device, host, and storage hit
attribution. For deployment-faithful results, copy `max_total_num_tokens=N` from
the real SGLang server startup log and launch the simulator with
`--max-total-tokens N`. Avoid relying on a separately estimated capacity when
comparing the simulator with production traces.

Supported predictors:

| Predictor | Purpose |
|---|---|
| `aisimulate` | Operator and module performance-database estimation. |
| `ml` | A trained sklearn-compatible 18-feature latency model. |
| `replay` | Exact or nearest-neighbor batch-composition replay. |

Relative predictor paths are resolved from the simulator configuration location.
Environment variables in paths use `${NAME}` syntax.

## Collect baselines and train an ML predictor

Use [collect_baseline](src/sglang_simulator/tools/collect_baseline/README.md) to capture request traces
and synchronized batch steps from a real SGLang server. Then use
[train_ml_time_predictor](src/sglang_simulator/tools/train_ml_time_predictor/README.md) to train an
18-feature latency model. Installing the package provides `sglang-simulator-collect`
and `sglang-simulator-train`, including a small collected training example.

## Workload formats

The Autobench trace format uses timestamps in milliseconds:

```json
{"prompt":[1,2,3],"prompt_len":3,"output_len":1,"timestamp":200}
```

Random and ShareGPT workloads are also supported by the runner API and serving
benchmark paths.

## Validation

Run the CPU compatibility and unit tests from the repository root, with the
`python/sglang` package importable from the same checkout (see Installation)
and one estimator extra installed:

```bash
pip install -e "tools/sglang-simulator[aisimulate]"
pip install pytest
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_runner.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_serving.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_cache_hit_ratio.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_offline_blocking.py
python3 -m pytest -q tools/sglang-simulator/test/test_aisimulate_predictor.py
python3 -m pytest -q tools/sglang-simulator/test/test_hardware_policy.py
python3 -m pytest -q tools/sglang-simulator/test/test_memory_hardening.py
```

Run each file as a separate pytest command because the runner test installs
process-global simulator hooks and state.

The Linux memory gates require a 10x logical KV-capacity change and a 100x
logical transfer-size change to add less than 10 MiB RSS, require repeated
request/abort/timeout rounds to remain within a 16 MiB steady-state RSS band,
and audit startup for model-weight file opens.

Run repository checks before submitting:

```bash
git ls-files -z tools/sglang-simulator | \
  xargs -0 env SKIP=no-commit-to-branch pre-commit run --files
```

Runtime changes should also be validated in a matching official SGLang image
with both `OFFLINE` and `BLOCKING` modes. Predictor changes should report
step-level error, and scheduler or cache changes should compare request latency,
throughput, and prefix-cache reuse against measured traces.
