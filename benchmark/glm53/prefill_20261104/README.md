# GLM5.3 prefill 单项优化记录

沿用 2026-11-04 已实测的原镜像基线 B0，不将 MQA beta 或候选 FlashMLA 自动并入基线。
原始结果保存在 `/public/home/maxiao1/scripts/glm5.3/fp8/runs/ifb_ab_20261104_1548/`。
本目录记录来源于已有实测文件；本轮代码接入尚未运行 DCU 测试。

## 固定基线

p2c1+p2c9，`sglang_glm_mx`，TP8/CP8/EP8，CP v2 interleave。
112640 输入、107008 缓存、5632 新输入、输出 1，BS8，每轮 32 请求，chunk32768，EAGLE5/1/6。
原命令快照见 `serve_baseline.sh.txt` 和 `bench_bs8.sh.txt`；包哈希见 `baseline.json`。
快照按原样保存，旧脚本顶部的节点示例注释不作为实际节点依据。

| ID | 唯一变化 | 平均 TTFT ms | 新输入 tok/s | 状态 |
| --- | --- | ---: | ---: | --- |
| B0 | 原镜像 | 5011.77 | 8724.79 | 已实测 |
| P01 | 原 LightOp 包，`LIGHTOP_MQA_LOGITS_BETA=1` | 4802.42 | 9036.83 | TTFT -4.18% |
| P02 | 仅候选 FlashMLA，beta 关闭 | 4973.28 | 8734.40 | TTFT -0.77%；吞吐收益未超过波动 |

P01 没有换 LightOp 包。P02 只通过隔离 PYTHONPATH 加载候选 FlashMLA，未覆盖镜像原包。
候选源码 `5c7f4a68d6e6c044eec8962d06e5fbc84ebc50ed`，扩展 SHA256
`aa1488929d25d525bd87b19528fa5e4f3b0f76406eaff9f6681151124604db3c`。
历史正式数据为两轮固定种子对照；历史 profile 种子不同，仅支持同 shape 比较。
随机 token 输出不是模型准确率验收，保留原报告的这一限制。

## 新增单项，全部待测

每行都单独与 B0 比较，其他新增开关保持 0；不要依次累加开启。

| ID | 仅开启项 | 实现范围 / 前置条件 |
| --- | --- | --- |
| P03 | `SGLANG_GLM_PREFILL_CP_OVERLAP=1` | 主 KV AllGather 与 Q absorb BMM 重叠；支持 CP v1 round-robin 和基线 CP v2 interleave，保留逻辑 token 截断与 padding；不是 Q projection overlap |
| P04 | `SGLANG_GLM_PREFILL_DIRECT_Q=1` | BF16 absorb BMM 写最终 Q buffer，避免拼接复制；保持权重与 scale，排除 DCP/LoRA/graph 等不兼容路径 |
| P05 | `SGLANG_GLM_PREFILL_NORM_QUANT=1` | Q RMSNorm + per-token FP8 quant，Q projection 消费预量化输入，BF16 Q 仍供 indexer；不包含 residual norm，不适用 AITER shuffled weight |
| P06 | `SGLANG_GLM_PREFILL_NATIVE_KV=1` | LightOp native prefix KV gather/dequant；需 NaN 兼容能力标记，缺失时报错，不能静默换旧 ABI |
| P09 | `SGLANG_GLM_PREFILL_INPUT_NORM_QUANT=1` | input/residual RMSNorm + FP8 量化；类型化显式传递给 fused QKV-A，BF16 保留给 indexer；要求声明匹配 FP32 residual 语义的 LightOp 包，当前 B0 包不满足 |
| P08 | `SGLANG_GLM_PREFILL_SHARED_QUANT=1` | DeepEP GLM shared expert 的 SwiGLU + per-token FP8 量化，交给 down projection 消费；普通 eager prefill，channelwise 动态 FP8；需独立数值和性能验证 |
| P07 | `SGLANG_HCU_MEGA_MOE_RUNTIME=deepgemm_mega` + MegaMoE backend | 接入 DeepGEMM 的合并版 runtime，保留 checkpoint FP8 权重与 SwiGLU；仅 gfx938、eager、同宿主 HIP IPC；当前两节点基线暂不能直接使用 |

P03–P06 限 GLM 普通 prefill。每项必须确认日志和 trace 命中，未命中标记“不适用”，不能填写性能收益。
P07 是独立实验入口，不是本次两节点 DeepEP 的已验证替换。跨节点 IPC、在线 EPLB receive stats 尚不支持。
如果必须改成单宿主或更新 DeepGEMM 包，需要独立的匹配对照，不能把拓扑/包差异计为 P07 单项收益。

基线包不提供 P05/P06 所需接口时，先记录“包能力缺失”。确需升级则隔离安装，并以同一新包、开关关闭作为该单项的匹配对照，同时保留相对 B0 的结果。不能全量更新三个算子库后再声称单项收益。

## 仍需继续实现 / 核验

- KV RMSNorm + Q/K RoPE + quant/store 大融合：现有接口涉及 CP 全局 cache location 和 Q head stride，尚未在此分支接入，不能直接启用 decode 的融合路径。
- residual/input norm + quant：框架接入 P09 已补齐，旧 LightOp residual 累加的 BF16 舍入差异仍需兼容算子解决；当前 B0 包尚不满足要求。
- shared-expert SwiGLU/quant 已新增独立开关 P08；overlap 先审计已有路径，普通 prefill 的数值/性能仍待验证。不复制 Spark GELU/INT8 模型分支。
- 历史 KV 下一层预取、indexer overlap、TopK reuse：代码中已有部分路径；先核对基线是否已启用，再决定是否能作为新的单项。
- FlashMLA / MQA beta 已有实测，不重复记成新优化；GEMM 选优按用户要求暂不处理。

## 每项记录格式

每项新建独立运行目录，保留 `command.sh`、`manifest.json`、服务日志、原始 benchmark JSONL、核验结果、profile 和分析。
manifest 包含：对照 ID、实际 SGLang commit/dirty patch、所有 rank 的包路径/.so 哈希、节点/设备、模型与映射哈希、全部环境变量、客户端版本、种子、唯一变化、开始结束时间。

先固定已有 B0 的脚本与包。源代码分支基点不是“已测容器版本”的证明：候选部署前保存容器源码快照并确认本次差异，若有无关框架漂移需单独记录，不能归为优化收益。
性能正式跑测不启 profiler；每轮按原协议清缓存、预热共享前缀，使用相同种子，保留逐请求 input/cached/output 核验。
profile 单独采集，并对齐同一主模型 32K chunk；同时报告整请求 TTFT 和 chunk critical path，不能把跨 stream kernel dur 简单相加。
每项保留原始结果，不覆盖旧数据；单项验证后另建组合实验 ID，再决定是否合入旧分支。
