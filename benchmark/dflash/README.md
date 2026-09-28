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
- Start at memory fraction 0.85 (override with `MEM_FRACTION_STATIC`), since a
  six-layer complete draft and its workspace replace the old one-layer MTP.
  Inspect actual pool capacity before restoring the previous concurrency/context.

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
