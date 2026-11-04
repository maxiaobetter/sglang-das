# 运行环境与优化覆盖核验

2026-11-04，在启动 P03 前重新核对 p2c1、p2c9 的 `sglang_glm_mx`。
两台容器均未发现活跃测试进程，8 张卡 VRAM used 均为 2 MiB。此为启动前快照，不代表持续保留资源。
所有核验均有本地原始证据，位于原工作区 `profiles/glm53-prefill-opt-20261104/`。

## 实测源码与包

两台完整 Python 源码 SHA256 清单一致，3451 个 `.py` 文件中：

- 3449 个与开发分支基点 `086ff10241` 完全相同。
- `srt/mem_cache/kv_cache_configurator.py` 是容器已有差异；候选直接复制容器基线源码，保留该文件，未用仓库版本覆盖。
- `_version.py` 是安装包生成文件，仓库基点未跟踪。

因此本轮使用“容器源码副本 + 本分支改动文件”的方式隔离加载；不能直接把整个仓库覆盖容器。
原 site-packages 保留不变，完整基线副本、完整候选副本和清单存放在：

```text
/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515/
├── baseline_source/sglang/
├── candidate_source/sglang/
├── installed_source_sha256.json
├── candidate_commit.txt
├── patched_files.json
├── serve_control.sh
├── serve_P03.sh
└── P03/
```

候选快照 commit 为 `9b7870b450624414da1524cc497a60521fd55bab`。只通过 PYTHONPATH 选择副本。

| 包 | 容器实际版本 | 与 B0 的关系 |
| --- | --- | --- |
| Torch | 2.11.0+dtk2604.2610211125.ga346d2 | 沿用 |
| LightOp | 0.6.0+dtk2604.torch2110.2610211335.ge564f5 | `.so` 哈希与 B0 一致 |
| FlashMLA | 1.2.0+dtk2604.torch2110.2610191020.g1287a0 | `.so` 哈希与 B0 一致 |
| DeepGEMM | 2.1.0+dtk2604.torch2110.2610202354.g9ab224 | `.so` 哈希与 B0 一致 |

完整哈希见 `baseline.json`，实际能力探测见 `runtime_audit.json`；不以远端分支最新状态代表已安装版本。

## 当前逐项结论

| 项目 | 静态核验与实际适用性 | 验证状态 |
| --- | --- | --- |
| P03 主 KV AllGather / Q absorb BMM overlap | CPv2 interleave 已接入，保持 logical/physical padding；配置只启用此新增开关 | 已启动隔离服务；未取得性能结果 |
| P04 直写 Q buffer | 新增开关独立，保持 BF16 absorb 权重/scale | 待独立验证 |
| P05 Q RMSNorm + FP8 quant | 基线有 `rms_norm_fp8_quant`，尚需验证归约/量化结果及消费者调用 | 待独立验证 |
| P06 native KV gather/dequant | 基线有 kernel 符号，但缺少 `PREFILL_GATHER_TRITON_NAN_COMPAT_VERSION`，不能只凭符号开启 | 基线包不满足接入契约，需兼容候选包及匹配对照 |
| residual/input norm + quant | 基线无 `RMS_NORM_FP8_FP32_RESIDUAL_WIDTHS` 能力声明；旧实现的 residual 舍入差异仍需解决 | 框架扩展和兼容算子尚未完成 |
| KV norm/RoPE/store 融合 | 基线有 q_cat 接口，但无 `MLA_ROPE_SUPPORTED_Q_HEAD_STRIDES` 声明；CP local/global cache loc 和 GLM Q stride 必须适配 | 尚未完成，不能将 decode 融合直接启用到 prefill |
| MegaMoE | 新 adapter 对应 develop 的 `deepgemm.mega`，不是容器原有 `deepgemm.op` 接口 | 同宿主 IPC 入口，跨节点可用性尚未证明 |
| 历史 KV 下一层预取 | 分支已有 `maybe_prefetch_next_full_attention_kv`，依赖 layer split；B0 的 IFB layer split 关闭 | 需作为独立配置实验，不能记作新实现收益 |
| shared expert overlap | `forward_deepep` 已有 alt_stream + event 路径；需核对 B0 的实际 stream/trace | 不重复宣称为新增实现 |
| P08 shared expert SwiGLU + quant | 已新增 DeepEP 普通 prefill 的独立开关及 down projection 预量化消费，需核对 BF16 中间舍入与 FP8 scale | 已接入，待独立数值和性能验证 |
| indexer overlap / TopK reuse | 实现已存在多个门控分支，需结合实际 B0 kernel/stream 确认 | 待覆盖核验 |

P03 与 Q projection 的 overlap 是不同候选，不合并记账。P01 MQA beta 和 P02 FlashMLA 的已有实测保持独立，见 README 和 results.csv。
任何仅完成静态检查、启动加载或符号探测的项目，均不能填写“正确性通过”或“性能提升”。
