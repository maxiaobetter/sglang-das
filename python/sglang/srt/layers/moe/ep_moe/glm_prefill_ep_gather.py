"""Default-off dispatch for the captured GLM FP8 prefill K8 gather kernel."""

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.glm_prefill_native import lightop_glm_op
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)


def try_glm_prefill_ep_gather_k8(
    input_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> bool:
    if not (
        envs.SGLANG_GLM_OPT_PREFILL.get() or envs.SGLANG_GLM_PREFILL_EP_GATHER_K8.get()
    ):
        return False
    if not has_forward_context():
        return False
    backend = get_forward_context().attn_backend
    # Published by metadata initialization from the actual ForwardBatch. It is
    # cleared on every metadata entry, including graph and speculative paths.
    if not getattr(backend, "_glm_prefill_ep_gather_k8_eligible", False):
        return False
    if (
        is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    tensors = (input_tensor, topk_ids, topk_weights, input_index, output_tensor)
    if not all(t.is_cuda and t.device == input_tensor.device for t in tensors):
        return False
    if torch.version.hip is None:
        return False
    arch = torch.cuda.get_device_properties(input_tensor.device).gcnArchName
    if arch.split(":", 1)[0] != "gfx938":
        return False
    if len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors):
        return False
    if not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors):
        return False
    if (
        input_tensor.dtype != torch.bfloat16
        or output_tensor.dtype != torch.bfloat16
        or topk_ids.dtype != torch.int64
        or topk_weights.dtype != torch.float32
        or input_index.dtype != torch.int32
        or input_tensor.ndim != 2
        or output_tensor.ndim != 2
        or topk_ids.ndim != 2
        or topk_ids.shape[1] != 8
        or topk_weights.shape != topk_ids.shape
        or input_index.shape != topk_ids.shape
        or output_tensor.shape[0] != topk_ids.shape[0]
        or input_tensor.shape[1] != output_tensor.shape[1]
        or input_tensor.shape[1] != 6144
        or input_tensor.shape[0] == 0
        or output_tensor.shape[0] == 0
    ):
        return False
    # The caller is exclusively the FP8 contiguous method's existing LightOp
    # no-map branch. Keep the supplied output and ordered FP32 K0..7 FMA math.
    lightop_glm_op("ep_gather", "specialize_k8")(
        input_tensor,
        topk_ids,
        topk_weights,
        input_index,
        None,
        output_tensor,
        specialize_k8=True,
    )
    return True
