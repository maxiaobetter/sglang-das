# GLM-5.3 Channel-FP8 + DFlash2 on HCU

Initial implementation based on `glm5.3-fp8` at `ea1c297dc23343cbe5c6b619f3094325fb5fe45b`.
This is a hardware-validation candidate; local lint is not serving, accuracy,
acceptance-rate or performance validation.

## Implemented configuration

- Target: local `hygon/GLM-5.3-Channel-FP8-w8a8` checkpoint, FP8 target KV.
- Draft: local `incoai/GLM-5.3-DFlash2` checkpoint, BF16 weights/KV, Triton attention,
  block size 8. Use the same checkpoint revisions and tokenizer on P and D.
- P: TP8/CP8/EP8, interleave, DSA cache LayerSplit, attention TP1.
- D: TP32/DP32/EP32 across two 16-device hosts, attention TP1, DP LM head.
- Real Mooncake transfer; PP1. DP support in this implementation is scoped to
  disaggregated decode, not mixed prefill/decode serving.

Each attention rank constructs a complete TP1 draft. Its model construction,
attention backend, KV projection, forward and selector run in a scoped topology;
the target retains its original topology. Draft KV is replicated across P CP
ranks. Existing LayerSplit ownership sends it from the last CP rank. This costs
more P memory than a sharded draft; KV budgeting accounts for the replicated
geometry. CP-v1 auxiliary hidden outputs are gathered back into token order;
CP-v2 already gathers these outputs in its wrapper.

Draft execution and scheduling are synchronous initially. An idle rank skips
draft work but participates in target collectives through eager execution.
Active ranks retain target CUDA graphs when their graph checks pass. DeepEP
configurations that keep only local token counts may mix active graph execution
with idle eager execution; this needs hardware validation.

Transfer managers retain the target topology and LayerSplit settings from
initialization, so background Mooncake transfers do not see draft overrides.

The optional Mooncake registration extension carries a target/draft configuration
signature and per-entry page sizes. The sender checks these during peer
registration. An incompatible peer fails its transfer rooms through the existing
registration-error path without terminating the transfer worker.
MLA target entries and all draft K/V entries have distinct IDs. Both P and D
must use this implementation; an old P sender cannot enforce the new contract.
The signature checks configuration, not checkpoint file contents.

## Launch

### Two-host P/D pair: 126 Prefill, 127 Decode

The two standalone scripts below use the paths verified inside each host's
`sglang_mx_glm53` container. Both default to the DFlash implementation checkout at
`/home/maxiao/GLM53/fp8/sglang-das` and prepend its `python` to `PYTHONPATH`.

| Host | Script | Worker | Devices | HTTP endpoint |
| --- | --- | --- | --- | --- |
| 10.41.101.126 | `prefill_cp8ep8_126.sh` | P: TP8/CP8/EP8 | 0-7 | `http://10.41.101.126:30033` |
| 10.41.101.127 | `decode_dp16ep16_127.sh` | D: TP16/DP16/EP16 | 0-15 | `http://10.41.101.127:30026` |

This is P/D disaggregation over Mooncake. Each worker has its own distributed
process group (`--nnodes 1`); P and D are not ranks of one TP group. Prefill's
bootstrap port is 8998. Point the P/D router at the two HTTP endpoints above and
that bootstrap port.

Inside the corresponding containers:

```bash
# 126: Prefill only
bash /home/maxiao/GLM53/fp8/sglang-das/benchmark/dflash/prefill_cp8ep8_126.sh

# 127: Decode only
bash /home/maxiao/GLM53/fp8/sglang-das/benchmark/dflash/decode_dp16ep16_127.sh
```

The scripts also work when copied to `/home/maxiao/GLM53/fp8/scripts/`. Target and
draft paths default to `/home/models/` on 126 and `/models/` on 127. They are
overridable through `MODEL_PATH` and `DRAFT_PATH`; use matching checkpoint
revisions on both sides. Logs go to `$BASE/logs/dflash_prefill_cp8ep8` and
`$BASE/logs/dflash_decode_dp16ep16`.

Use `DRY_RUN=1 bash <script>` to check file paths and print the command without
starting a server. The initial memory fractions are 0.90 for P and 0.85 for D.
P failed KV budgeting at 0.85 after loading target and draft weights on this
CP8 setup (the logged minimum was about 0.852). These are starting settings;
validate the resulting KV capacity and forward workspace on the actual workload.
Both scripts use DFlash block8,
BF16 draft KV, FP8 target KV, and Decode BS5 per DP rank (80 total requests).
DeepEP capacity64 covers the resulting 40 verify tokens per rank. Recalculate
dispatch capacity when changing the per-rank batch size; extra CLI arguments
are forwarded unchanged. These scripts clear simulated acceptance, the skipped
DP synchronization and inherited static-LP probabilities.
The paired mask-aware sparse MQA/TopK optimization is disabled for this baseline.

Both launchers enable the existing GEMM tuning artifacts by default:

```bash
BLAS_TUNING_DIR=/home/maxiao/GLM53/fp8/ep32_optimization/blas_tuning/final
export HIPBLASLT_TUNING_OVERRIDE_FILE="$BLAS_TUNING_DIR/hipblaslt.config"
export ROCBLAS_TENSILE_LIBPATH="$BLAS_TUNING_DIR/library_gpu6"
```

Override `BLAS_TUNING_DIR` when moving the scripts, or set `USE_BLAS_TUNING=0`
for an explicit untuned comparison. The launchers check that both artifacts
exist. The saved manifest binds the tuning to gfx938, 64 CU and specific BLAS
library hashes; revalidate after changing those libraries. Its original EP32
fake-prefill validation does not establish real P/D acceptance or accuracy.

`eagle516_prefill_cp8ep8_126.sh` and `eagle516_decode_dp16ep16_127.sh` provide
the corresponding EAGLE 5/1/6 comparison, with Decode draft LM-head VP16 and
the same GEMM tuning, model, KV dtype, memory fractions and request limits.
EAGLE uses its supported overlap scheduler; DFlash currently disables it.

### Original three-host example: Prefill plus two Decode hosts

Use the existing working HCU image and its LightOp, FlashMLA, DeepEP, ROCSHMEM,
network and tuning environment. Deploy this checkout on every node and set
`SGLANG_HOME` to it. Do not reuse a PYTHONPATH that puts the frozen delivery ahead
of this checkout; the launcher prepends the selected checkout itself.

Set these per host before launching:

```bash
export SGLANG_HOME=/path/to/this/checkout
export MODEL_PATH=/path/to/GLM-5.3-Channel-FP8-w8a8
export DRAFT_PATH=/path/to/GLM-5.3-DFlash2
export LOCAL_IP=<this-host-ip>
export NCCL_SOCKET_IFNAME=<this-host-interface>
export IB_DEVICES=<comma-separated-assigned-NICs>
export HIP_VISIBLE_DEVICES=<comma-separated-assigned-devices>
```

Allocate eight devices to P and sixteen devices on each D node. Do not overlap
P/D device allocations. Then run:

```bash
# P host, eight devices:
bash benchmark/dflash/launch_glm53_hcu_pd.sh prefill

# Each D host, sixteen devices; identical DIST_INIT_ADDR, NODE_RANK=0 or 1:
export DIST_INIT_ADDR=<D-rank-0-ip>:5001
export NODE_RANK=0
bash benchmark/dflash/launch_glm53_hcu_pd.sh decode
```

Connect the P/D endpoints through the existing P/D router. The launcher only
starts one server instance and does not modify running services or the router.
It accepts extra server arguments after the stage, including existing static
EPLB/LPLB options. The static probabilities environment variable can be retained.

Differences from the supplied MTP launchers:

- Replace EAGLE 5/1/6 with DFLASH 1/1/8 and the separate draft path.
- Remove `--speculative-draft-lm-head-vp-size 16`: it is an EAGLE option.
- Replace fake transfer with Mooncake. Unset simulated acceptance and
  `SGLANG_SCHEDULER_SKIP_ALL_GATHER`; independent requests require real DP counts.
- Use BF16 draft KV while retaining FP8 target KV.
- Raise DeepEP dispatch capacity from 32 to 64 for 5 x 8 verify tokens per rank.
  Recalculate it when increasing per-rank batch size or block size.
- Start at memory fraction 0.90 for P and 0.85 for D (override with
  `MEM_FRACTION_STATIC`). The six-layer complete draft replaces the old one-layer
  MTP, and target plus draft weights must fit inside the static budget before
  KV can be allocated. Inspect actual pool capacity and forward workspace before
  restoring the previous concurrency/context.

## 限量 debug 与 CPU 离线分析

Debug 默认关闭。以下步骤供之后重新分配节点时使用，本次修改没有运行节点实测。
启用后会同步复制小批量张量到 CPU 并写盘，**只能用于正确性诊断，不能用该次运行计性能**。
文件含请求的 token、隐藏状态和概率，请保存在用于排障的目录中。

启动 P、D 前分别设置独立的输出目录；不要让两侧覆盖同一目录，也不要混放不同实验：

```bash
# P 进程启动环境
export SGLANG_DFLASH_DEBUG_DIR=/path/to/debug/run-cp8/prefill
# D 进程启动环境使用另一个目录：
# export SGLANG_DFLASH_DEBUG_DIR=/path/to/debug/run-cp8/decode

export SGLANG_DFLASH_DEBUG_RID_PREFIX=dflash-debug-
export SGLANG_DFLASH_DEBUG_MAX_REQUESTS=2
export SGLANG_DFLASH_DEBUG_MAX_STEPS=16
export SGLANG_DFLASH_DEBUG_MAX_TOKENS=9
```

这些是**每进程**的上限，CP8 的各 rank 都可能生成文件，不能把上限理解为整个服务
最多两份文件。`MAX_STEPS` 按请求和记录阶段分别计数，`MAX_TOKENS` 限制每次采样的
位置数；每个已采样位置保留完整的 feature/head 向量。仅匹配 RID 前缀的请求被记录。
若达到限额，缺失记录不能视为检查通过。关闭时取消 `SGLANG_DFLASH_DEBUG_DIR`。

建议生成一份 `/generate` 请求文件，在 CP8 和 no-CP 对照中复用相同 `input_ids`。
下面只加载本地 tokenizer，不加载模型权重。`MODEL_PATH` 使用本地 target 模型路径，
`ROUTER_URL` 指向现有的 P/D router：

```bash
export MODEL_PATH=/path/to/GLM-5.3-Channel-FP8-w8a8
export ROUTER_URL=http://<pd-router-host>:<port>
python - <<'PY' > /tmp/dflash-debug-intro.json
import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    os.environ["MODEL_PATH"], trust_remote_code=True, local_files_only=True
)
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请简要介绍你自己。"}],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=False,
)
if isinstance(input_ids, dict):
    input_ids = input_ids["input_ids"]
print(json.dumps({
    "rid": "dflash-debug-intro",
    "input_ids": input_ids,
    "sampling_params": {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 128},
    "stream": False,
}, ensure_ascii=False))
PY
curl --fail-with-body "$ROUTER_URL/generate" \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/dflash-debug-intro.json
```

请求使用的 RID 必须保留到 worker；若 router 改写了 RID，先确认改写后的前缀是否仍
匹配。no-CP 对照需要一套能运行相同模型的合法并行配置，不能只删 CP 参数而保留
不匹配的 TP/EP 参数。两次实验复用同一请求文件、checkpoint 和采样参数。

将 P/D 输出目录复制到本地后，用装有 PyTorch 的 CPU Python 环境分析：

```bash
# 同一次运行的 P/D KV 和 D 端验收诊断
python benchmark/dflash/analyze_dflash_debug.py \
    /local/debug/run-cp8/prefill /local/debug/run-cp8/decode \
    --json /local/debug/run-cp8/report.json

# 另一次运行保持完全相同的输入 token；比较 CP8/no-CP 的 prefill aux
python benchmark/dflash/analyze_dflash_debug.py \
    /local/debug/run-cp8/prefill /local/debug/run-cp8/decode \
    --compare /local/debug/run-no-cp/prefill \
    --json /local/debug/cp-comparison.json
```

CLI 只执行 `torch.load(..., map_location="cpu", weights_only=True)`，不加载 checkpoint、
tokenizer 或 SGLang runtime。它汇总以下证据：

- **每个请求的 draft 第 1 位**独立显示实际接受次数/到达次数、平均 `alpha`、
  `Cmass`、`tail_q`；第 2 位及以后只统计前缀接受后真正到达的位置。
  `alpha = sum(min(p_C, q))`，`Cmass = sum(p_C)`，
  `tail_q = sum(q[p_C == 0])`。它们分别表示给定该位置分布的理论接受概率、
  target 在候选集合内的概率质量、draft 落到 target 零概率位置的概率质量。
- 只有 `p_source=actual_kernel_input` 且 `q_source=actual_kernel_input` 才是实际
  verifier 输入的证据。若标记 `reconstructed_after_accept`，指标只是重构参考，
  不能据此断言 kernel 使用了这些概率。实际 uniform 可用时另行报告逐位决定和
  前缀接受长度的复核差异；截断后的前缀长度会按已记录位置数比较。
- P/D draft KV 按 `bootstrap_room`/RID、逻辑位置、层名对齐，并检查该位置输入 token
  与完整原始 `input_ids` 哈希一致，报告 `exact`、`max_abs`、`mean_abs` 与非有限差值数。物理 `cache_locs` 不参与
  比较；只比较已采样位置，不能证明未采样页正确。
- `--compare` 的 prefill aux 只按完整原始 `input_ids` 的 SHA-256 和逻辑位置匹配，
  再检查输入 token 一致，按 `capture_layers` 拆分报告逐层误差。不会仅因形状相同就
  配对；不同 CP rank 的副本分别列出。FP8 下非零误差仍需结合数值规模判断。
- 输出会明确列出配对缺失、快照失败、非法文件或张量；不会把无数据当成一致。
  完整 JSON 保留每个比较的源文件、rank、dtype 和位置，便于追查。
- 同时显示 `p_sum`/`q_sum` 范围，以及实际 dense q 总质量和候选内 q 总质量的最大差值。
  重复候选、非有限或负概率行从理论均值中排除，实际接受次数仍保留；若质量不归一化
  或候选外还有 q 残留，`alpha` 不能解释为完整有效分布的接受概率。
  文件阶段计数还包含 `verify_state`：它保存已提交的 verify 输入对应 aux/KV，供定位
  后续计算；默认 P/D 比较只用完整 prefill 与 D 首轮接收快照，不混比不同生成 token。

这是限量观察，不提供统计显著性的结论；16 个 step 的实际接受率与理论均值出现差异，
本身不能证明实现错误。先看同一批实际 `p/q/uniform` 的验收复核、KV 传输和 aux
一致性，再决定是否需要增加采样量。切勿用开启 debug 时的延迟和吞吐评价性能。

## Hardware validation still required

1. Check imported SGLang path, target/draft checkpoint config and startup pool
   capacities on both sides; confirm BF16 draft and FP8 target KV.
2. Run real P/D short prompts, then prompts spanning multiple prefill chunks.
   Cover uneven DP load, idle ranks, completion/retraction and prefix-cache hits.
3. Compare greedy output and task accuracy with target-only serving. FP8 target
   quantization may change draft acceptance; it does not require FP8 draft weights.
4. Record actual accepted tokens per target forward, TPOT, throughput and peak
   memory at the same workload/concurrency as the existing MTP baseline.
5. Validate target graph replay separately from eager execution. Draft graph
   capture is intentionally disabled in this initial CP/DP path.

No new kernels or unit tests are introduced by this implementation.
