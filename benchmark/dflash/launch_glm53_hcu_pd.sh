#!/usr/bin/env bash
# Use the HCU runtime environment that already serves the target checkpoint.
set -euo pipefail

STAGE="${1:?Usage: bash launch_glm53_hcu_pd.sh prefill|decode [extra server arguments]}"
shift
SGLANG_HOME="${SGLANG_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${MODEL_PATH:?Set MODEL_PATH to the local GLM-5.3 Channel-FP8 directory}"
: "${DRAFT_PATH:?Set DRAFT_PATH to the local Inco GLM-5.3-DFlash2 directory}"
: "${LOCAL_IP:?Set LOCAL_IP to this host address}"
: "${IB_DEVICES:?Set IB_DEVICES to the NICs assigned to these devices}"
: "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME to this host network interface}"
: "${HIP_VISIBLE_DEVICES:?Explicitly select the HCU devices for this instance}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
test -f "$MODEL_PATH/config.json"
test -f "$DRAFT_PATH/config.json"
test -d "$SGLANG_HOME/python/sglang"
export PYTHONPATH="$SGLANG_HOME/python${PYTHONPATH:+:$PYTHONPATH}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export MC_TE_FILTERS="$IB_DEVICES"
export MC_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
# Independent DP requests need real sync counts, including idle ranks.
unset SGLANG_SCHEDULER_SKIP_ALL_GATHER
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

COMMON=(
    --model-path "$MODEL_PATH" --trust-remote-code
    --served-model-name GLM-5.3-Channel-FP8-w8a8
    --host "$LOCAL_IP" --dtype bfloat16
    --context-length "${CONTEXT_LENGTH:-1048576}"
    --page-size 64 --kv-cache-dtype fp8_e4m3
    --dist-timeout 10000 --watchdog-timeout 3600
    --moe-dense-tp-size 1 --moe-a2a-backend deepep
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_PATH"
    --speculative-draft-attention-backend triton
    --speculative-draft-kv-cache-dtype bf16
    --speculative-num-steps 1 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 8 --disable-overlap-schedule
    --disaggregation-mode "$STAGE"
    --disaggregation-transfer-backend mooncake
    --disaggregation-ib-device "$IB_DEVICES"
)
case "$STAGE" in
    prefill)
        STAGE_ARGS=(
            --port "${PORT:-30033}"
            --dist-init-addr "${DIST_INIT_ADDR:-$LOCAL_IP:5033}"
            --nnodes 1 --node-rank 0
            --tp-size 8 --ep-size 8 --attn-cp-size 8
            --enable-prefill-cp --cp-strategy interleave
            --enable-dsa-cache-layer-split --deepep-mode normal
            --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv
            --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}"
            --chunked-prefill-size 32768 --max-prefill-tokens 32768
            --max-running-requests 96 --disable-cuda-graph
            --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8998}"
        )
        ;;
    decode)
        : "${DIST_INIT_ADDR:?Set the same rank-0 address:port on both D nodes}"
        : "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
        # BS5 x block8 = 40 verify tokens/rank; the old MTP capacity32 is too small.
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
        STAGE_ARGS=(
            --port "${PORT:-30024}" --dist-init-addr "$DIST_INIT_ADDR"
            --nnodes 2 --node-rank "$NODE_RANK"
            --tp-size 32 --dp-size 32 --ep-size 32
            --enable-dp-attention --enable-dp-lm-head --deepep-mode low_latency
            --dsa-prefill-backend flashmla_auto --dsa-decode-backend flashmla_kv
            --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}"
            --chunked-prefill-size -1 --cuda-graph-max-bs 5
            --max-running-requests 160
        )
        ;;
    *) echo 'Stage must be prefill or decode' >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" -m sglang.launch_server "${COMMON[@]}" "${STAGE_ARGS[@]}" "$@"
