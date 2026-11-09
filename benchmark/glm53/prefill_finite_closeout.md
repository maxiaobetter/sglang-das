# GLM 5.3 Channel-FP8 framework optimizations

The model covered by this work is GLM-5.3-Channel-FP8-w8a8. The branch name
`glm5.3-w4a8` does not change the model's W8A8 quantization contract.
All framework options below default to off. Set them before importing SGLang.

```bash
# Prefill workers
export SGLANG_GLM_OPT_PREFILL=1
export LIGHTOP_MQA_LOGITS_BETA=1
# Optional, separate SGLang dispatch + DeepGEMM kernel dependency:
export SGLANG_GLM_PREFILL_K1_INDIRECT_A=1

# Decode workers
export SGLANG_GLM_OPT_DECODE=1
# Temporary independent hcu_ops path, to be retired after BLAS integration:
export SGLANG_GLM_HCU_GEMM=1
```

`SGLANG_GLM_OPT_PREFILL` selects the retained framework paths: KV/RoPE store,
shared-expert overlap, gate K1 vectorization, packed KV direct store, physical
KV compact, EP gather K8, AddRMS align16, and the two inverse-broadcast quantizers.
The existing per-operator shape, layout, quantization, phase, and graph guards
still apply. Unsupported inputs use the original path. Existing fine-grained
framework flags remain available for diagnosis.

`SGLANG_GLM_OPT_DECODE` selects the existing MLA fusion and draft concat routing,
and scoped o_proj quantization. The o_proj path admits BF16 K=16384, E4M3FN
output, FP32 token scales, local M=1..5 in DECODE or M=6/12/18/24/30 in
TARGET_VERIFY/DRAFT_EXTEND_V2, on gfx938. It cannot intercept other projections.
The capability whitelist for norm+quant remains in place: upstream LightOp
master does not advertise `RMS_NORM_FP8_FP32_RESIDUAL_WIDTHS`; those calls keep
the existing norm/quant path. This PR does not add that capability marker.

Neither framework switch enables LightOp's beta MQA setting, DeepGEMM
indirect-A, skinny GEMM/BMM, BLAS tuning, routing changes, or experimental
page-table fusion. LightOp environment variables retain their original scope.

## Operator delivery and dependencies

The matching LightOp PR exports `GLM53_OPT_ABI_VERSION=1` and these explicit
native APIs (ordinary LightOp APIs and their defaults are unchanged):

| Framework caller | Installed LightOp entrypoint | Retained operand scope |
| --- | --- | --- |
| RMSNorm.forward_hip helper | glm53_prefill_add_rms_align16 | BF16 H6144, M704/832/1408/3520/4096, aligned16, distinct buffers, FP32 residual sum |
| DeepEP FP8 contiguous gather helper | glm53_prefill_ep_gather_k8 | BF16, H6144, K8, no expert map; ordered FP32 K0..7 FMA |
| per-token quantizer | glm53_prefill_token_quant_inverse | BF16 4096x16384, E4M3FN, FP32 per-token scales |
| unclamped expert activation helper | glm53_prefill_activation_inverse | BF16 H4096 to H2048; M14336/84224/84992; original scale floor |
| MLA o_proj quantizer | glm53_decode_token_quant_broadcast | BF16 K16384; phase/M/layout checks above |

These five APIs remove four experimental SO path variables and the EP gather
PYTHONPATH dependency. They are lazy-loaded from the installed LightOp module;
missing or incompatible declared APIs raise errors. There is no runtime JIT,
library replacement, or installation step in a serving worker.

Indirect-A has code in both repositories. SGLang builds route metadata and
invokes the first expert GEMM; DeepGEMM supplies the compatible native kernel.
Its independent flag is intentionally not included in the prefill switch.
This submission keeps the frozen `r09plus_wasp_indirect` dependency and its
native SHA/Torch/HIP/DeepEP/DeepGEMM checks. Install that module on PYTHONPATH,
or configure the existing `SGLANG_GLM_PREFILL_K1_INDIRECT_A_SO` path. It is not
provided by the LightOp PR; DeepGEMM source is organized locally and not pushed.
The existing LightOp B2 identity or the new explicit GLM ABI is admitted;
this loader change has not been exercised on GPU.

Skinny GEMM remains `SGLANG_GLM_HCU_GEMM=1`, with its separate `hcu_ops` package
and `skinny_cfg.json`. Its hooks cover FP8 GEMM, MLA BF16 BMM and BF16 LM head.
Untuned operands keep DeepGEMM/torch operations. The historical wrapper SHA is
`ed2807c5589a074a4f137a788a8fc6f8a5ad571a633d7fc37d4813530abb0284`,
native SHA `8aa3f2e39bf93dfca597714d80511f355cb6b600a5b09d0c0fdd86c83fac5e34`,
and configuration SHA `be50e87f4d0d6664bbb0d65865406d8dabb5c3f95a8494fcd8eb163b52860339`.
Its kernel source is not part of DeepGEMM indirect-A or this LightOp submission.

## Evidence and outstanding alignment

This is a draft source delivery, not a claim that the rebuilt native package
or this PR head has passed GPU accuracy validation. No service was changed or
started during source organization. Per the operator-alignment requirement,
full model accuracy and performance reruns must wait for the outstanding
installed-package and same-input checks.

| Path | Evidence state | Remaining work |
| --- | --- | --- |
| Eight retained prefill candidates | Frozen candidate002 real-input/dispatch evidence; scoped numerical replays recorded | Rebuilt LightOp and new SGLang head same-input checks; loaded-object identity gaps |
| Prefill model graph | Unverified; retained candidates exclude capture | Do not infer graph acceptance from eager evidence |
| Decode quant broadcast | Frozen agg1 15 numeric cases and three phase GPU witnesses | Formal installed entrypoint and new context plumbing replay, including changed-input graph |
| Decode MLA/concat | Historical runtime/profile evidence, partial checks | Full mathematical/installed-version alignment and changed-input graph replay |
| Skinny GEMM/BMM | BS1 historical kernel evidence; all 32 BS4 ranks showed no skinny kernels | Same-input numerical alignment and actual shape-by-shape coverage |
| Norm+quant | Capability absent on upstream master | Keep fallback; no capability added without proof |
| Page gather+repeat6 | Performance-only archived run; numerical alignment unverified | Source retained separately; excluded from these switches and this runtime patch |

Prefill historical performance refers to frozen candidate002, not this PR
head: two fixed prefix-hit seeds measured TTFT down 4.97%-5.12%, with model
graph disabled. Heldout32 A/B answers matched; long-token equivalence remains
unverified. Decode fake-prefill and simulated acceptance performance is not
P/D accuracy evidence. The native package port changes symbol namespaces,
bindings and packaging; it has not been rebuilt or validated on HCU here.

Local validation for this source delivery uses the repository CI pre-commit
configuration, Python syntax parsing and `git diff --check`. No new unit tests
or full model runs are included; compile and CI checks run on GitHub.
