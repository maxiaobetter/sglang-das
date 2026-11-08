# GLM5.3 FP8 prefill finite closeout (draft integration)

This ports the 15 framework files from the measured, isolated
`final_closeout_candidate_source_002` onto `glm5.3-w4a8` after PR #54
(base `59186a4d253606444eaf57aca40858d976bc92e1`). It is a draft integration,
not a complete operator distribution or a full-model precision release.
All new optimization switches default off. No GPU/model test has been run
on this PR revision: its changes from the measured sources are explicit
extension-path configuration, documentation, and project lint formatting.

## Scope and native fallback

- Gate K1 vectorization preserves the original multiply order.
- Packed KV store copies opaque FP8 cache bytes after the existing CP collective.
  It does not bypass padding or replace the collective/backend/allocator.
- Physical KV compaction shares only row metadata across layers, preserves sparse
  index slots/order/sentinels, and dequantizes each layer's own cache rows.
- EP gather K8 preserves ordered FP32 accumulation; Add RMS retains the frozen
  BF16/residual/normalization contract; indirect K1 keeps the returned routing
  maps through K2 and gather.
- Token inverse is limited to contiguous, zero-offset BF16 `[4096,16384]` and
  native FP8 E4M3FN/per-token FP32 scales. Other widths, including K12288, use native.
- Expert activation inverse is limited to the no-limit/no-topkcount caller,
  BF16 H4096 to FP8 H2048, with M in `{14336,84224,84992}`. Other shapes use native.

The shared backend scope uses the actual effective ordinary EXTEND batch,
prefix presence, main GLM model, interleave CP, and compatible cache/layout.
It clears its token on each metadata entry. Individual dispatchers additionally
check their own shape, dtype, alignment, ownership, quantization and ABI.
The measured inverse/Add RMS/indirect paths require TP/CP/EP8. Decode, MTP,
CUDA graph, compiler and TBO exclusions remain in place; flags alone do not
establish eligibility. Inverse dispatch also requires the shared scope to be
published by one of physical compact, EP gather, Add RMS or indirect K1.

Unsupported inputs/disabled flags take the existing native paths. Explicitly
opting into an eligible external kernel with a missing binary, wrong SHA or ABI
raises the existing validation error; it does not silently load another binary
or switch quantization/math. The path options below change binary location only,
not accepted hashes or the kernel's Python module name/ABI.

## Explicit external dependencies

These diagnostic extensions are **not bundled or built by this PR**. Obtain the
exact already-validated binaries from the measured deployment. Arbitrary rebuilt
binaries are rejected even if they have matching filenames. The measured ABI is
Torch `2.11.0`, HIP `6.3.26113`, CXX11 ABI enabled, gfx938. Add RMS, indirect K1 and
inverse helpers also enforce their pinned operator-library identities.

| Feature enable (set to `1`) | Binary configuration | Required SHA256 |
|---|---|---|
| `SGLANG_GLM_PREFILL_ADD_RMS_ALIGN16` | `SGLANG_GLM_PREFILL_ADD_RMS_ALIGN16_SO` = absolute path to `add_rms_align16_diag.so` | `11e6a0dff8d96913fb6a22ff12adee7179c004a3dab97c2443fa2bbb0ecab295` |
| `SGLANG_GLM_PREFILL_K1_INDIRECT_A` | `SGLANG_GLM_PREFILL_K1_INDIRECT_A_SO` = absolute path to `r09plus_wasp_indirect.so` | `e59ac60daff9ff6788bf05c91e48dbf4cf3e966d41f2079d6e652be4b30cac9d` |
| `SGLANG_GLM_PREFILL_TOKEN_QUANT_INV_BROADCAST` | `SGLANG_GLM_PREFILL_TOKEN_QUANT_INV_SO` = absolute path to `token_inverse_diag.so` | `e7870a15db717c3761e80725e238b8709ee1adb4ff92239121b7e4b8211f0361` |
| `SGLANG_GLM_PREFILL_ACTIVATION_INV_BROADCAST` | `SGLANG_GLM_PREFILL_ACTIVATION_INV_SO` = absolute path to `activation_inverse_diag.so` | `a13d3751d5cde8e9a4b66ea1cca769431dbb6051a1ef113bab6ef7b129d4bf9a` |
| `SGLANG_GLM_PREFILL_EP_GATHER_K8` | Put `epgather_k8_diag` extension on `PYTHONPATH` | `e225959f6b88a57af897471241ae994ed24d1e1de48b6a0ef07c0383df6f638d` |

Pinned operator dependencies: LightOp
`f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c`,
DeepGEMM `79325b382a56de60f69cfaf8f75fe2b725d2092c002d1cf67bd1248f1537af7a`,
DeepEP `7ee4f4edec4e6fa275b6d64d842e3925c4ac2ca4b72419046a03d0d103a16d8d`.
Physical compact also needs the measured FlashMLA raw sparse-prefill interface
with preserved order and topk lengths; a sorting wrapper is rejected.
The measured FlashMLA SO hash was
`9627d7abd2dd5d9d4e48a1f820ee415ce541534c0e5fc4b00e86f8066b02dfcc`.
These hashes describe the deployment, not source-branch versions or a claim that
all installed loaded objects were proven identical (see limitations below).

The measured combination additionally enabled
`SGLANG_GLM_PREFILL_GATE_K1_VECTOR`, `SGLANG_GLM_PREFILL_PACKED_KV_DIRECT_STORE`,
`SGLANG_GLM_PREFILL_PHYSICAL_KV_COMPACT`, and the already-existing
`SGLANG_GLM_PREFILL_KV_ROPE_STORE` / `SGLANG_GLM_PREFILL_SHARED_OVERLAP`.
It used `LIGHTOP_MQA_LOGITS_BETA=1`, the prior PR's matching LightOp/FlashMLA setup,
and channel-FP8 W8A8 (not W4A8 or block quantization). This is not a general
launch recipe for other installations. Optional once-per-rank/shape inverse logs
use `SGLANG_GLM_PREFILL_INVERSE_TRACE=1`; Add RMS/indirect traces have their own
`SGLANG_GLM_PREFILL_ADD_RMS_ALIGN16_TRACE` / `SGLANG_GLM_PREFILL_K1_INDIRECT_A_TRACE`.
Set all new optimization flags to `0` or unset them for native fallback.

## Frozen deployment results (not PR-head GPU validation)

Two nodes, four GPUs each, TP/CP/EP8, chunk32768, model CUDA graph disabled,
EAGLE 5 steps/topk1/draft6. Each seed used 32 requests, batch/concurrency8,
input112640/cache107008/output1, with identical flush/warmup. Per seed, compare
B to `(A1+A2)/2`; TTFT gain is `1-B/bracket`, throughput gain `B/bracket-1`.

| Seed | A1 TTFT ms | B TTFT ms | A2 TTFT ms | TTFT reduction | New-input throughput gain | A2/A1 TTFT drift |
|---|---:|---:|---:|---:|---:|---:|
| 2026110401 | 4547.83 | 4317.46 | 4553.43 | 5.12% | 5.29% | +0.123% |
| 2026110402 | 4567.17 | 4340.92 | 4569.01 | 4.97% | 5.03% | +0.040% |

B throughput: 10085.47 / 10034.66 new-input tok/s. These are two fixed workloads,
not a confidence interval or a sum of kernel microbenchmark gains.

- GSM20 smoke: A1 17/20, B 18/20; first five questions overlap examples.
- Disjoint heldout32: both 30/32; all 32 answers/predictions/prompt hashes/finish
  reasons/completion counts matched. A2 did not repeat either short evaluation.
- Two random 65536-token inputs, 32 output tokens: all A1/B/A2 sequences differ;
  both A1/A2 baseline pairs diverge from the third token. **Long-token accuracy is
  UNVERIFIED**: baseline instability prevents isolating candidate influence.
  No labels exist for these random inputs.
- Actual ordinary EXTEND logs showed token/expert inverse on all eight ranks.
  MTP DRAFT/DECODE/TARGET_VERIFY all showed correlated native GPU work, with both
  inverse candidates, Add RMS candidate and K1 init/map absent. DECODE did not
  show the expert-specific native symbol, so that component's fallback is unproven.
- Shared-path FlashMLA/LightOp/inverse library maps had matching device/inode/hash.
  Container-installed DeepGEMM/RCCL/libtorch_hip mapped-object identity remains
  **UNVERIFIED** because overlay backing devices differed and `map_files` was
  inaccessible. Disk-path hashes were not substituted as loaded-object proof.

Dense SiLU+FP8 fusion, shared K2048, other token widths, packed Q reuse, CP
padding bypass and alpha-add fusion are excluded. Dense intermediates matched,
but native GEMM itself did not reproduce historical captured outputs. No new
math, relaxed tolerances, captures or GPU tests were added for this port.
Model graph-enabled paths, peak model memory and complete 120-operator coverage
were not validated. Timed output1 stores decoded text, not output token IDs.

Frozen full A manifest: `89142f93eaf187f340ccc24d3623e89f157dfb506d22df9e5f781d7c67cb4e4c`.
Frozen full B002: `bf3deadde9729f1ea2cf759d0d9d74e585e4cae8c46b83093cf732d64a94be23`.
Original 15-file patch: `118112582e327ac06b0aa77a8aae0f7b4f594f281143b1e3452e61e9a72f3095`.
Final measured report: `69474596685bdde876c67b0dc62158bfa7123556255a41ba2de8f45b94ac347c`;
result JSON: `4240a6c78138ba98fc8642684596501145773720c2ef04f38e7d3686c820ad0a`.
Large profiles, binaries and input archives remain in the frozen deployment
artifacts and are not included in this framework PR.
