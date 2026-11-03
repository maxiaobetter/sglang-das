# GLM5.3 LightOp fusion test branch

This branch adds only MLA preprocessing and RMSNorm/FP8 quantization fusion. It does not add selective GEMM, skinny kernels, draft VP communication changes, or new BLAS tuning.

Base revision: `086ff10241` (the existing GLM/DCU working baseline, not origin/main).

## Package

Use the validated Python 3.10 / Linux x86_64 / DTK 26041 / Torch 2.11 environment.

Wheel: `lightop-0.6.0+das.dtk26041-cp310-cp310-linux_x86_64.whl`

SHA256: `20ca56d3fb80b70ab5007c32d340d00981912c3e02e3b4012175fd94edeee183`

Cluster source: `/public/home/maxiao1/scripts/glm5.3/fp8/ab_20261029/stride256_20261030/build_source/lightop/dist/`

```bash
sha256sum lightop-0.6.0+das.dtk26041-cp310-cp310-linux_x86_64.whl
python3 -m pip install --no-deps --force-reinstall ./lightop-0.6.0+das.dtk26041-cp310-cp310-linux_x86_64.whl
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

Verify the imported SGLang source points to this checkout and the LightOp extension supports Q head strides 192/256 and norm-quant FP32 residual semantics. Existing runtime overlays can shadow this branch.

## Compare OFF / ON

Keep the same wheel, model, EPLB/LPLB, DeepEP, BLAS settings, batch size, and scheduler settings in both runs. Restart the server when changing switches.

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_GLM_LIGHTOP_FUSIONS=1
export SGLANG_LIGHTOP_NORM_QUANT=1
export SGLANG_LIGHTOP_MLA_KV_FUSED=1
export SGLANG_LIGHTOP_FUSION_PHASES=verify
```

For OFF, set only `SGLANG_GLM_LIGHTOP_FUSIONS=0`. Use the same existing server command in both runs. The default fusion scope is target verify; draft decode/extend remain unchanged. Unsupported layouts or missing capabilities retain the original path, so an enabled flag does not prove kernel execution.

Expected ON trace kernels: `rms_rope_kernel`, `rms_norm_fp8_quant_`, and `rms_norm_fp8_quant_residual`. Existing SiLU/mul/FP8 fusion is part of the baseline, not a new optimization here.

## Validation scope

Historical wheel numerical checks and kernel-hit evidence exist. This extracted branch receives project CI-equivalent lint and Python syntax checks; no new GPU or full-model accuracy run is claimed. Compare model accuracy separately from fake-prefill performance. No new unit tests are added.
