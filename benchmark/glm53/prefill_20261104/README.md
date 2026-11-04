# GLM5.3 prefill 单项优化记录

沿用 2026-11-04 已实测的原镜像基线 B0，不将 MQA beta 或候选 FlashMLA 自动并入基线。
原始结果保存在 `/public/home/maxiao1/scripts/glm5.3/fp8/runs/ifb_ab_20261104_1548/`。
历史 B0/P01/P02 保留原结果；本轮 fresh control 和 P03-r1 已完成 DCU 实测，P03-r2 已完成独立验证。

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

## 新增单项，按项验证

每行都单独与 B0 比较，其他新增开关保持 0；不要依次累加开启。

Fresh control 平均 TTFT **5019.78 ms**，与历史 B0 相差 +0.16%；P03-r1 为 **10533.73 ms**，较 fresh control 高 **109.84%**，保持关闭。P03-r2 只修订通信 stream 复用后为 **4986.79 ms**，比 fresh control 低 **0.66%**，尚不能确认稳定收益、保持默认关闭。P04 两轮为 **4992.98 ms**，相对fresh control低 **0.53%**，同样保持默认关闭。详见 `P03.md`、`P04.md` 和 `control_results.json`。

扩展框架源码 `69fbccd6ed` 的全部开关关闭对照已完成：TTFT 5062.01/5022.37 ms，均值 **5042.19 ms**，GSM8K 19/20。三库沿用 B0，8 rank 源码与哈希核验、8 trace 完整性通过。P11/P12 使用它作同源码匹配对照；历史 B0 和原源码 fresh control 保留，见 `framework_control_results.json`。

P11单项已完成：TTFT均值 **4962.62 ms**，相对framework-control低 **1.58%**；shared stream与routed区间相交102.51ms，整步低42.15ms，不能将相交量直接记为收益。GSM8K19/20，仍为小样本/两轮结果，详见 `P11.md`。

P12单项也已完成：TTFT均值 **5015.97 ms**，比framework-control低 **0.52%**，比历史B0高 **0.084%**；Q projection并行命中，32K step只低5.16ms，未确立稳定收益，见 `P12.md`。

LightOp B2全部关闭的匹配对照已完成：TTFT **5017.45 ms**，19/20，32K chunk **3272.20 ms**；换包本身未确立加速。后续融合项分别与此组对照，见 `B2-control.md`。

P05单项已完成：TTFT **5024.89ms**，较同包关闭对照高 **0.15%**；78次融合命中，32K step基本不变，默认关闭，见 `P05.md`。

独立 `megamoe` 已补充为P13单独审计；当前normal接口声明支持GLM shape，但量化/路由时序、fabric和权重内存尚待核对。P07合并版排除结论不涵盖它，见 `P13.md`。

| ID | 仅开启项 | 实现范围 / 前置条件 |
| --- | --- | --- |
| P03 | `SGLANG_GLM_PREFILL_CP_OVERLAP=1` | 主 KV AllGather 与 Q absorb BMM 重叠；支持 CP v1 round-robin 和基线 CP v2 interleave，保留逻辑 token 截断与 padding；不是 Q projection overlap |
| P04 | `SGLANG_GLM_PREFILL_DIRECT_Q=1` | BF16 absorb BMM 写最终 Q buffer，避免拼接复制；保持权重与 scale，排除 DCP/LoRA/graph 等不兼容路径 |
| P05 | `SGLANG_GLM_PREFILL_NORM_QUANT=1` | Q RMSNorm + per-token FP8 quant，Q projection 消费预量化输入，BF16 Q 仍供 indexer；不包含 residual norm，不适用 AITER shuffled weight；要求 LightOp match_sglang ABI v1 |
| P06 | `SGLANG_GLM_PREFILL_NATIVE_KV=1` | LightOp native prefix KV gather/dequant；需 NaN 兼容能力标记，缺失时报错，不能静默换旧 ABI |
| P09 | `SGLANG_GLM_PREFILL_INPUT_NORM_QUANT=1` | input/residual RMSNorm + FP8 量化；类型化显式传递给 fused QKV-A，BF16 保留给 indexer；要求 LightOp match_sglang ABI v1，显式 opt-in 匹配 FP32 residual/quant 语义，当前 B0 包不满足 |
| P10 | `SGLANG_GLM_PREFILL_KV_ROPE_STORE=1` | prefix prefill 的 KV norm + Q/K RoPE + FP8 store + Q concat；CPv2 interleave 交换完整656-byte cache行并写全局loc，避免重复量化；原包无GLM Q stride能力声明，需兼容候选包；框架已实现、尚未实测 |
| P11 | `SGLANG_GLM_PREFILL_SHARED_OVERLAP=1` | 复用现有 DeepEP shared/routed 双stream组织，普通GLM prefill显式开启；全模型复用一条shared stream，保持decode原门控，独立于P08量化融合；已测，详见P11.md |
| P12 | `SGLANG_GLM_PREFILL_Q_PROJ_OVERLAP=1` | Q-B projection 与实际执行的DSA indexer重叠；独立compute stream，排除collective/LoRA，不复用indexer通信stream；不是P03的absorb BMM overlap；已测，默认关闭，见P12.md |
| P08 | `SGLANG_GLM_PREFILL_SHARED_QUANT=1` | DeepEP GLM shared expert 的 SwiGLU + per-token FP8 量化，交给 down projection 消费；普通 eager prefill、channelwise动态FP8；要求显式SGLang SwiGLU FP8 ABI v1，旧融合核scale下限不同，不能直接替换；待测 |
| P07 | 当前pin不纳入运行矩阵 | groupwise激活量化不符合本组per-token约束，新增adapter已从交付移除；证据及后续要求见P07.md |

P03–P06 限 GLM 普通 prefill。每项必须确认日志和 trace 命中，未命中标记“不适用”，不能填写性能收益。
P07当前pin已排除，见 `P07.md`：实际L1→L2使用groupwise激活scale，与本组固定GLM per-token语义不同；跨宿主HIP IPC也有独立失败证据。实验adapter保留在提交历史，当前交付移除新增runtime入口，不更改原有MegaMoE后端。未将它放入任何性能组合。

基线包不提供 P05/P06 所需接口时，先记录“包能力缺失”。确需升级则隔离安装，并以同一新包、开关关闭作为该单项的匹配对照，同时保留相对 B0 的结果。不能全量更新三个算子库后再声称单项收益。

## 仍需继续实现 / 核验

- KV RMSNorm + Q/K RoPE + quant/store 大融合已新增 P10 框架入口，尚未验证。限定有prefix、普通pool、CPv2 interleave，原始cold prefill继续使用未量化BF16 K；wrapped/compact/HiSparse pool保持原writer。P10消费已完成的Q absorb BMM，不能与P03同时打开；P10已包含Q concat，组合时P04无需重复开启。
- residual/input norm + quant：框架接入 P09 已补齐，旧 LightOp residual 累加的 BF16 舍入差异仍需兼容算子解决；当前 B0 包尚不满足要求。
- shared-expert SwiGLU/quant 为 P08；已有 shared/routed overlap 在B0受 `SGLANG_ROCM_USE_MULTI_STREAM` 门控、未开启。新增P11只为普通GLM prefill开启，保持其余模式原有门控，不能与P08合并记账。不复制Spark GELU/INT8模型分支。
- 历史 KV 下一层预取、indexer overlap、TopK reuse：覆盖核验见 `COVERAGE.md`。Indexer小窗口重叠和TopK复用已命中；LayerSplit仅支持PD prefill，不能直接加入本组IFB协议。
- FlashMLA / MQA beta 已有实测，不重复记成新优化；GEMM 选优按用户要求暂不处理。

## 每项记录格式

每项新建独立运行目录，保留 `command.sh`、`manifest.json`、服务日志、原始 benchmark JSONL、核验结果、profile 和分析。
manifest 包含：对照 ID、实际 SGLang commit/dirty patch、所有 rank 的包路径/.so 哈希、节点/设备、模型与映射哈希、全部环境变量、客户端版本、种子、唯一变化、开始结束时间。

先固定已有 B0 的脚本与包。源代码分支基点不是“已测容器版本”的证明：候选部署前保存容器源码快照并确认本次差异，若有无关框架漂移需单独记录，不能归为优化收益。
性能正式跑测不启 profiler；每轮按原协议清缓存、预热共享前缀，使用相同种子，保留逐请求 input/cached/output 核验。
profile 单独采集，并对齐同一主模型 32K chunk；同时报告整请求 TTFT 和 chunk critical path，不能把跨 stream kernel dur 简单相加。
每项保留原始结果，不覆盖旧数据；单项验证后另建组合实验 ID，再决定是否合入旧分支。
