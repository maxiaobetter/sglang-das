# GLM R09 prefill configuration

This change extracts only the R09 KV producer/packed CP exchange and shared-expert overlap from the FP8 performance branch. Both framework flags default to false. The beta MQA implementation belongs to LightOp, not this repository. Q-projection overlap, CP overlap, direct-Q, experimental norm/activation quantization and MegaMoE are excluded.

Source `env.sh` after the existing server environment. No new CLI arguments are required. Retain the tested DeepEP NORMAL, eager, TP8/CP8/EP8, interleave CP and chunk32768 configuration. The fused KV path requires an ordinary prefix prefill with the guarded BF16 activation/RoPE and 656-byte FP8 cache layout. Other paths fall back. Shared overlap is limited to ordinary GLM prefill with separate shared experts and without SBO/TBO or graph capture. Neither path changes the expert weight quantization.

Use the matching LightOp B2 build from c3881932743ed63e2fd6c30c8474d46ac8d5c15a (DTK26041, Python3.10, gfx938); KV fusion requires its Q-head-stride-256 support. Keep the tested FlashMLA/DeepGEMM/DeepEP packages. Merely importing a new package is not runtime validation.

## Existing evidence, not a benchmark of this PR

R09 used the channel-FP8 model, input112640/cached107008/output1, BS8,32requests,chunk32768,p2c1+p2c9:

| Configuration | Mean TTFT ms | GSM20 | 100 questions |
| --- | ---: | --- | --- |
| beta + KV fusion + shared overlap | 4699.16 | 19/20 | 96/100 |
| beta reference after R09 | 4787.75 | 19/20 | 97/100 |
| R09 + Q projection overlap (excluded here) | 5031.86 | 19/20 | 96/100 |

R09 improved TTFT by about1.85% against the closing beta reference. The confirmation run did not capture a fresh profile. These measurements do not validate W4A8, other BS, PD, decode, MTP or graph execution on the target branch.

## Extraction and alignment status

- Source implementation: FP8 optimization snapshot7684f58399ad70de47de7678897f7ea5aab6107e, KV commit ca56102ac6 and shared overlap commit c3343455c0.
- KV producer retains the same LightOp call, parameters, scale epsilon, output layout and CP gather/store. Its prerequisite checks remain; the unrelated experimental CP-overlap flag is absent.
- Shared overlap retains original expert math and the base branch's `_forward_shared_experts(hidden_states)` signature. Only stream selection/lifetime/join changes. Its private eligibility marker is named for overlap rather than the excluded shared quant experiment.
- Common eligibility/stream/report helpers are extracted without Q-projection or other experimental implementations.
- Target branch runtime alignment: **unverified**. Before enabling this extracted code in a different installed environment, capture the required native profiles on actual request-serving ranks and compare the full dispatch/kernel chain, installed package hashes/ABI, shape/stride/dtype, quantization/rounding and cache/index layouts. Close discrepancies with the same actual inputs and intermediate/output comparisons before whole-model performance tuning. Include MTP/graph replay if those paths are later enabled.

Only repository lint, syntax and diff checks are performed for this extraction. No new unit tests, package installation, GPU run or service changes are included. Keep this PR a draft until target-branch alignment is complete.
