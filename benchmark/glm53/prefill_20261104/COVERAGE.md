# DSpark prefill 优化覆盖核验

核验对象是固定 B0 的 GLM5.3 Channel FP8 普通 IFB prefill，不能把 Spark 的 INT8、GELU、W4A8 或不同拓扑收益直接搬入。框架引用本分支基点 `086ff10241` 与不可变候选 `69fbccd6ed`；DSpark 参考 `021a69ba2392`，LightOp 参考 `cf5b95407474`。以下“已有”与“待测”严格分开。

| DSpark 方向 | GLM 对应项 | 当前结论 / 后续验收 |
| --- | --- | --- |
| 主 KV AllGather 与 Q absorb BMM overlap | P03 | r1每层新stream导致回归；r2复用一条stream恢复到基线附近。单项正确性、两轮压测与同种子32K trace齐全；尚无稳定端到端收益，默认关闭 |
| Q projection 与 indexer overlap | P12 | 单独接入非collective Q-B投影，与P03不同；GSM8K19/20、两轮TTFT低0.52%，32K step仅低5.16ms，未确立稳定收益，默认关闭 |
| absorb Q 直写最终 buffer | P04 | 78次full concat改为tail copy，strided BMM略慢；两轮TTFT小幅变化，保持关闭 |
| Q norm + FP8 quant | P05 | 需要显式match_sglang ABI，保留indexer消费的BF16 Q；B2的16组norm/quant原生输出/scale逐位一致；模型/性能待验 |
| input/residual norm + FP8 quant | P09 | 需保持FP32 residual累加、norm BF16中间舍入和无scale下限；独立入口已接入，B2原生含residual对照通过，模型/性能待验 |
| KV norm + Q/K RoPE + quant/store | P10 | 适配GLM Q stride256和CP全局cache loc；交换含scale/rope的656-byte行，cold BF16路径保留；B2原生stride256的168+2组通过，框架/模型待验 |
| prefix KV gather/dequant替换 | P06 | B2的数值对照通过，但原有速度断言失败，237.10us vs Triton145.93us；保持关闭，模型收益未验 |
| MegaMoE route/GEMM/combine | P07排除 | 实际L1→L2的groupwise scale不符合GLM per-token约束，新增adapter移除、原有后端保留。双宿主HIP IPC另有失败证据，不能据此认定HyLink硬件不支持；兼容ABI及未实现工作见P07.md |
| shared激活 + FP8 quant | P08 | GLM使用SwiGLU，旧融合核scale下限不同，新增显式兼容入口；框架与原生实现已提交，8组原生输出/scale逐位一致，模型/性能待验 |
| shared/routed expert overlap | P11 | B0未开ROCM多流门控；新增普通prefill限定入口，复用一条stream。两轮TTFT低1.58%，GSM8K19/20，32K step低42.15ms；稳定收益尚未确立，见P11.md |
| 下一层历史Index-K/Main-KV预取 | 已有LayerSplit路径 | 基线LayerSplit关闭，trace无broadcast。当前server_args明确只允许PD prefill，普通IFB还会执行decode，不能只加flag或删除守卫来比较。PD需要独立协议，不计作本次IFB已验证优化 |
| Indexer LayerNorm/RoPE融合 | B0已有 | 主32K chunk实际21次opt_layernorm_rope_split，累计4.43ms，不重复实现/计收益 |
| Index-K gather与query计算重叠 | B0已有 | 21次非主stream AG；21次query Hadamard使用[131072,128]×[128,128] GEMM，累计1.95ms，与AG区间相交1.44ms。已存在小窗口重叠，不等于P03或P12；相交时间不是A/B节省量 |
| Q/K量化、gate scale及Index-K写入融合 | B0已有FP8实现 | 实际21次fused_qk_quant_weight_mul_and_store_index_k_cache，累计1.47ms；不能因新版库含同名能力就重记新增 |
| Hadamard/prepare大融合 | LightOp现有Q/K prepare不匹配 | q_prepare输出INT8且M<=36；k_prepare为INT8缓存且M<=20。本组FP8、每rank4096新token，不能直接替换。BF16 Hadamard已有缓存128阶矩阵，未每层重建；进一步FP8大融合需独立接口和舍入核验，未声称完成 |
| 另一个FP8 Norm/RoPE/Q-quant融合核 | LightOp `fuse_norm_rope_quant_fp8` | `csrc/fuse_layernorm_rope_gourpquant_fp8.cu`的Q分支在RoPE后直接量化，没有Hadamard，scale还含1e-12下限；GLM原路径在BF16 Hadamard后才做FP8量化，不能把该核直接换入或删除Hadamard。只取K的norm/RoPE部分会重复B0已存在的融合，不构成新增收益 |
| TopK复用/避免重复排序 | B0已有 | 模型78层中21个full、57个shared，实际只有21次indexer预处理；MQA/TopK各126次是21层×6个budget分块，不能当成126层。旧NSA环境名在当前EnvBoolWithAlias仍有效。trace无sort命名kernel，但这不足以证明kernel内部无排序；不更改模型的共享策略 |
| route激活/量化与topk gate融合 | B0已有部分 | routed MoE已有75次act_and_mul_fp8_quant及75次moe_fused_gate_kernel_gourp1；shared的BF16 SiLU+单独量化仍由P08处理，不搬Spark GELU/topk14常量 |
| sparse attention kernel与索引布局 | P02及已有路径 | 78次FlashMLA sparse累计777.81ms，是热点；候选FlashMLA已独立测过，微小变化未确立稳定吞吐收益，不自动替换B0。排序/复用保留原语义，不能据INT8 profile直接估算FP8收益 |
| MQA kernel替换 | P01 | 原包只开beta已有独立结果，TTFT下降4.18%；组合待测，不自动并入B0 |
| 不必要copy与metadata重建 | P04及审计 | P04消除Q full concat；V absorb直写、Hadamard矩阵缓存、跨层TopK复用原已存在。P10处理KV重复写入/交换；其他copy须给出调用、shape及存活周期后再改，不能按copy总量直接宣称可省 |

## 原始证据和边界

原工作区 `profiles/glm53-prefill-opt-20261104/control/coverage_trace.json` 保存kernel计数、exact-shape与External id关联；`model_structure.json` 保存实际模型结构。对齐 `step[EXTEND bs=6 toks=32768]`，客户端正式BS8，GPU step3272.25ms。跨stream累计耗时不能直接加成wall time或收益。

Mega原生扩展SHA256 `8e900c9894fea6b080ec3c3946d41a7fdfbede8151a39e9ef36f400096dcff18`，基线deepgemm.op保留原哈希。双宿主probe仅验证2MiB uncached分配/64-byte句柄交换和打开，不是MoE正确性或性能测试；两端在open阶段都报invalid device pointer。命令与结果保留于远端本轮目录 `mega_hip_ipc_probe/`、本地 `mega-build/ipc_probe/`。未改阈值，未新增仓库单测。

LightOp B1 `8c2b40a818fd`已编译、wheel ZIP完整性通过；B2 `c3881932743e`包含SwiGLU显式兼容入口，已完成编译、ZIP及原生数值检查，P06性能断言失败，模型验收待执行。编译成功不代表原生数值正确、模型准确率通过或性能提升。仍待验收的适用项不得记作完成。
