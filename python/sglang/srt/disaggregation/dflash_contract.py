"""DFlash P/D registration checks, carried in the optional metadata extension."""

import hashlib
import json


def configure_dflash_transfer(kv_args, scheduler):
    if not scheduler.spec_algorithm.is_dflash():
        return
    worker = scheduler.draft_worker
    if not worker.draft_parallel.enabled:
        return
    runner = worker.draft_model_runner
    config = runner.model_config.hf_config
    pool = runner.token_to_kv_pool
    target = scheduler.model_config.hf_text_config
    # Paths differ across hosts. Compare model geometry and quantization,
    # not local path spelling; operators must deploy identical weight revisions.
    signature = {
        "target_architectures": scheduler.model_config.hf_config.architectures,
        "target_layers": target.num_hidden_layers,
        "hidden_size": target.hidden_size,
        "vocab_size": target.vocab_size,
        "target_kv_dtype": kv_args.kv_cache_dtype_str,
        "target_kv_layout": kv_args.kv_cache_layout,
        "target_quantization": getattr(target, "quantization_config", None),
        "draft_architectures": config.architectures,
        "draft_layers": config.num_hidden_layers,
        "draft_heads": config.num_attention_heads,
        "draft_kv_heads": config.num_key_value_heads,
        "draft_head_dim": getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        ),
        "draft_config": config.dflash_config,
        "block_size": worker.block_size,
        "draft_window_size": worker.draft_window_size,
        "draft_kv_dtype": str(pool.dtype),
        "page_size": pool.page_size,
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    kv_args.v4_transfer_metadata = {
        "version": 1,
        "dflash": {"version": 1, "signature": digest},
        "kv_sizes": list(kv_args.kv_item_lens),
    }


def validate_dflash_transfer(src_metadata, dst_metadata, src_ids, dst_ids):
    src = (src_metadata or {}).get("dflash")
    dst = (dst_metadata or {}).get("dflash")
    if src is None and dst is None:
        return
    if src is None or dst is None or src != dst:
        raise RuntimeError(
            "DFlash P/D contract mismatch: deploy the same implementation, "
            "target/draft config, block size and draft KV dtype on both sides"
        )
    src_sizes = src_metadata.get("kv_sizes")
    dst_sizes = dst_metadata.get("kv_sizes")
    if not isinstance(src_sizes, list) or not isinstance(dst_sizes, list):
        raise RuntimeError("DFlash P/D requires KV page sizes on both peers")
    if src_ids is None or dst_ids is None:
        raise RuntimeError("DFlash P/D requires KV entry ids on both peers")
    if len(src_ids) != len(src_sizes) or len(dst_ids) != len(dst_sizes):
        raise RuntimeError("DFlash P/D requires complete KV entry ids and sizes")
    if len(set(src_ids)) != len(src_ids) or len(set(dst_ids)) != len(dst_ids):
        raise RuntimeError("DFlash P/D requires distinct target and draft K/V ids")
    destination = dict(zip(dst_ids, dst_sizes))
    for layer_id, size in zip(src_ids, src_sizes):
        if destination.get(layer_id) != size:
            raise RuntimeError(
                f"DFlash P/D KV page layout mismatch at entry {layer_id}"
            )
