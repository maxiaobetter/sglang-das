#!/usr/bin/env bash
# Run inside 10.41.101.126 / sglang_mx_glm53. P side of the P/D pair.
set -euo pipefail

BASE="${BASE:-/home/maxiao/GLM53/fp8}"
SGLANG_HOME="${SGLANG_HOME:-$BASE/sglang-das}"
MODEL_PATH="${MODEL_PATH:-/home/models/GLM-5.3-Channel-FP8-w8a8}"
DRAFT_PATH="${DRAFT_PATH:-/home/models/GLM-5.3-DFlash2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_IP=10.41.101.126
PORT="${PORT:-30033}"
DIST_PORT="${DIST_PORT:-5033}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
LOG_DIR="${LOG_DIR:-$BASE/logs/dflash_prefill_cp8ep8}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp23s0u2c2}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"

# Import the DFlash branch before the container's installed SGLang.
test -f "$SGLANG_HOME/python/sglang/srt/speculative/dflash_parallel.py"
test -f "$MODEL_PATH/config.json"
test -f "$DRAFT_PATH/config.json"
export PYTHONPATH="$SGLANG_HOME/python${PYTHONPATH:+:$PYTHONPATH}"
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_FUSE_TOPK=1
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK=0
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1

export HSA_ENABLE_COREDUMP=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export DEEP_EP_NORMAL_MNVL=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/home/hqd/glm5.2/topo.config}"
test -f "$ROCSHMEM_TOPO_FILE_FORCE"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="$IB_DEVICES"
export MC_ALLOWED_IBV_DEVICES="$IB_DEVICES"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

# Real acceptance and transfer; EP32 maps/tuning are not defaults for this pair.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
unset SGLANG_SCHEDULER_SKIP_ALL_GATHER SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS
unset ROCBLAS_TENSILE_LIBPATH HIPBLASLT_TUNING_OVERRIDE_FILE

EXTRA_ARGS=()
if [[ -n "${DEEPEP_CONFIG:-}" ]]; then
    test -f "$DEEPEP_CONFIG"
    EXTRA_ARGS+=(--deepep-config "$DEEPEP_CONFIG")
fi
CMD=("$PYTHON_BIN" -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --trust-remote-code
    --served-model-name GLM-5.3-Channel-FP8-w8a8
    --host "$LOCAL_IP"
    --port "$PORT"
    # nnodes describes this P worker; the D worker has a separate process group.
    --dist-init-addr "$LOCAL_IP:$DIST_PORT"
    --nnodes 1
    --node-rank 0
    --tp-size 8
    --ep-size 8
    --attn-cp-size 8
    --moe-dense-tp-size 1
    --enable-prefill-cp
    --cp-strategy interleave
    --enable-dsa-cache-layer-split
    --moe-a2a-backend deepep
    --deepep-mode normal
    --dsa-prefill-backend flashmla_sparse
    --dsa-decode-backend flashmla_kv
    --context-length "${CONTEXT_LENGTH:-1048576}"
    --dtype bfloat16
    --dist-timeout 10000
    --watchdog-timeout 3600
    --page-size 64
    --kv-cache-dtype fp8_e4m3
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}"
    --chunked-prefill-size 32768
    --max-prefill-tokens 32768
    --max-running-requests 96
    --disable-cuda-graph
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_PATH"
    --speculative-draft-attention-backend triton
    --speculative-draft-kv-cache-dtype bf16
    --speculative-num-steps 1
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 8
    --disable-overlap-schedule
    --disaggregation-mode prefill
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
    --disaggregation-ib-device "$IB_DEVICES"
    --reasoning-parser glm45
    --tool-call-parser glm47
    "${EXTRA_ARGS[@]}"
    "$@"
)

printf 'SGLANG_HOME=%s\nP=%s:%s bootstrap=%s devices=%s\n' \
    "$SGLANG_HOME" "$LOCAL_IP" "$PORT" "$BOOTSTRAP_PORT" "$HIP_VISIBLE_DEVICES"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$LOG_DIR"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
