# sglang-model → sglang-das 代码同步记录

## 同步范围（2026-10-29）

本批次审查sglang-model自2026-09-15起、截至 `d3db5682d1560c4d707feb4d17a92edac1b3c91f` 的26个候选提交，目标分支为 `mx_int8/main_glm53_flash_feature_cleanup_tx`，同步前基线为 `1b0a352df3975baa40569aa4ff5d7caea11e1a6c`。范围按committer timestamp包含2026-09-15当天，仅限源上界可达提交。

9个源提交全部或部分迁移，4个已有覆盖或保留目标差异，13个因目标缺少对应功能而跳过。同步采用线性追加，保留目标已有功能与差异化实现，不合并两仓库原历史。

## 同步内容

- 增加HiCache host Mamba槽位使用量与总量指标。
- 增加默认关闭的fake-bootstrap radix缓存开关。
- 修复compressed-tensors预量化输入能力声明与消费，避免重复量化。
- 支持HCU ReplaySSM fp16 checkpoint，保留FP32 g/beta与累加。
- 增加可选TileLang target-verify实现，并合并padding、stride、fp16及回退安全修复。
- 使用PREBUILT pinned staging与异步H2D，调整allocator默认行为；补充Mamba descriptor异步H2D及回归测试。
- 调整uniform共享前缀请求为round-robin，保留ordered/Zipf及预热剔除规则。
- 补充失效prefill连接的manager socket与monitor清理，等待发送排空。
- 适配目标接口并补充CPU/GPU测试；具体源提交、跳过原因与目标提交见逐提交账本。

## 运行行为变化

| 项目 | 本次行为 / 保留项 |
| --- | --- |
| `SGLANG_FAKE_BOOTSTRAP_USE_RADIX` | 默认关闭；stub benchmark 可设 1。常规 warmup/health 路径仍默认不插 radix。 |
| `SGLANG_KDA_USE_TILELANG_VERIFY` | 默认关闭；HCU KDA ReplaySSM verify 可设 1，不支持的形状/布局或 host-side kernel 异常回退现有 Triton。 |
| KDA fp16 checkpoint | 参数 hook 的已有告警保留，HCU checkpoint/pool 的 fp32-only guard 已移除；g/beta 仍 FP32。kernel parity 已测，fp16 全服务精度尚未测。 |
| `SGLANG_KVALLOC_KERNEL` | 默认从 true 改为 false；显式设 1 仍可选择 native allocator。微基准及上述D-only整体A/B已测，未单独隔离allocator的整服务收益。 |
| generated-shared-prefix | 新生成的默认 uniform 请求按组 round-robin；保留 ordered/Zipf；旧缓存保留旧顺序。warmup 仍从正式统计剔除，数量必须小于总请求数。 |
| 共享前缀预热约定 | uniform、G 个组可使用 2*G 条不同 suffix 的预热，请保证还剩正式请求；目标保留按组大小分波发请求。缓存复用还取决于服务端配置，应从服务日志核实。 |

## 同步规范

1. 固定源上界、目标起点和分支后枚举候选。区分 author date 与 committer date，按源拓扑顺序分析依赖；跨仓库 SHA 不等不能当作未覆盖的证据。
2. 先找目标的同名/改名/拆分实现、调用者和测试，再判定迁移、部分迁移、已覆盖、跳过。文件路径不同不等于功能不存在；目标已主动删除的路径保持删除。
3. 适用补丁使用 `cherry-pick -x`；重构适配在提交正文记录完整 `Source-commits` 或源 SHA。依赖安全修复可以合并。追加提交保持线性，不穿插两仓库原历史。
4. 冲突按目标结构解决，保留目标独有模式、配置入口和数据格式；部分迁移逐项说明跳过内容。避免整文件采用源版本，尤其 GLM/DSA、encoder、runtime-context、KV allocator 和量化分派。
5. 每个源 SHA 在下表只有一条结论。跳过表示对应源路径未同步，不表示目标不存在同类问题；如果后续引入该功能，应重新审查相关跳过条目。
6. 区分静态/CPU/GPU/服务端验证。没有实测时不把源 commit message 中的性能或精度数字写成目标验证结论。

后续增量以本批次源上界为游标，先确认其为新源上界的祖先，再按拓扑顺序审查新增提交。若源历史重写，使用patch-id和实际差异重新对账。保留跳过记录，目标引入相关功能后重新评估。

## 测试与对比结果

### 功能验证

- 20个变更Python文件静态解析及差异空白检查通过。
- portable CPU测试9/9、完整package量化接口测试3/3、TileLang GPU测试8/8、descriptor GPU回归2/2、KV allocator数据一致性测试22/22通过。量化接口测试使用mock GEMM，不代表完整量化GPU数值验证。
- 两侧真实PD/HiCache的16并发验证通过，精度评测194个题目键一致。
- 主比较使用FP32 checkpoint，TileLang verify关闭。fp16与TileLang仅有局部GPU验证，未覆盖开启后的全服务精度；文字评测不覆盖多模态端到端路径。

对比基线为 `1b0a352df3975baa40569aa4ff5d7caea11e1a6c`；同步后快照为 `65c61d81cb38d0704c608a92871ca3d24559cce9`，两侧一致应用descriptor公共修复 `60f425ec3bd38bc338dea1ce181c178c5b24cb2a`，公共修复不计入同步收益。两侧固定模型、依赖、硬件、功能配置及同步前版本的压测客户端；allocator采用各版本默认行为，未做独立消融。

### 精度对比

采样参数一致：seed=42、temperature=1、top_p=0.95、max_tokens=64000；每侧各运行一次，不复用预测缓存。

| 数据集 | 同步前 | 同步后 | 变化 |
| --- | --- | --- | --- |
| AIME25 | 25/30（83.33%） | 26/30（86.67%） | +3.33个百分点 |
| HumanEval | 163/164（99.39%） | 164/164（100%） | +0.61个百分点 |

AIME25同步前失败的17、18、29转为通过：17为截断，18为望远镜乘积末项错误，29为极小值条件理解错误；同步前通过的13、19转为截断，12、14两侧均截断。同步后四个失败均达到64000输出token且最终答案为空；部分反复验算，未完成推导是否存在额外数学错误无法确定，不统称机械复读。HumanEval/116由排序实现错误转为通过，同步后无失败样本。

单次随机采样得分略高，不能据此认定确定性精度提升，也不能将变化归因于特定同步提交或量化误差。

### 性能对比

工作负载为D-only fake-prefill：输入81920 token、共享前缀79872 token、输出1024 token、并发128。每侧3轮，每轮384个正式请求，预热剔除；六轮均384/384成功，每轮生成393216 token。

| 指标 | 同步前 | 同步后 | 变化 |
| --- | --- | --- | --- |
| 平均输出吞吐（token/s） | 5286.50 | 5982.70 | +13.17% |
| 平均TPOT（ms） | 14.48 | 14.41 | -0.51% |
| 平均TTFT（ms） | 1139.19 | 1145.18 | +0.53% |
| P99 TTFT（ms） | 7734.78 | 7953.29 | +2.83% |
| P99 TPOT（ms） | 30.74 | 30.16 | -1.90% |
| P99 ITL（ms） | 67.06 | 68.90 | +2.75% |

表中指标为三轮结果的算术均值；P99为三轮各自P99的均值，不是合并请求后的分位数。

| 轮次 | 同步前吞吐（token/s） | 同步后吞吐（token/s） | 同步前TPOT（ms） | 同步后TPOT（ms） |
| --- | --- | --- | --- | --- |
| 1 | 5971.46 | 6166.38 | 14.09 | 14.13 |
| 2 | 4452.98 | 5801.30 | 14.55 | 14.13 |
| 3 | 5435.07 | 5980.41 | 14.81 | 14.97 |

两侧各16-rank算子profile全部解析成功，共80条rank/kernel/grid/block对齐记录。W4A8 GEMM平均单次耗时115.23→113.02µs，DeepEP dispatch 152.05→134.43µs、combine 194.37→198.15µs，KDA recurrent 156.431→156.423µs。KDA基本不变，dispatch逐rank变化范围为-35%至+69%，存在明显通信波动。

本轮观测到吞吐均值提高13.17%，平均TPOT与TTFT基本持平。吞吐样本标准差为770.06→182.55 token/s，两侧范围重叠；每侧三轮及短profile不足以将差异全部归因于同步代码，不能视为已证明的稳定收益。

## 逐提交账本

以下证据路径均相对于目标 `python/sglang/srt/`，以 `kernels/` 开头的除外；测试位于 `test/registered/`。目标 SHA 指本同步分支中的提交，可配合最后的验证修正提交阅读。

| 源完整 SHA | 原主题 | 结论 | 目标提交 | 依据与适配说明 |
| --- | --- | --- | --- | --- |
| `7b10e8c737b86634fe1d07b31856bb26c9b362bc` | fix(nsa): avoid Q quantization JIT specialization by row count | 跳过 | — | 目标 GLM kpool 没有 `_quantize_q_rows`；其 Q/索引量化实现不同。保留 `layers/attention/glm5_next/kpool/kernels.py`，不恢复源 INT8 Q kernel 或已删除测试。 |
| `b878794235c9a29b3c7c604d4ce95b671cb54bda` | perf: 融合 NSA/MoE 输入准备算子并减少 H2D/D2D 搬运 | 跳过 | — | 源 `SGLANG_ENABLE_RUNTIME_FAST_PATH`、decode_input/replay_metadata/input_prepare 路径已被目标 `38cf417d89` 明确删除；不重新引入。 |
| `76747d5adab2bdb077b2c6ebf26a4b97bed90f31` | fix(epd): release prefill RDMA buffer per-part to avoid pool exhaustion | 跳过 | — | 源旧 receiver 的“克隆每个 part 后立即释放 RDMA buffer”路径不存在；目标 `encoder/receiver.py:GlmWaitingRDMARequest` 将 GPU view 生命周期绑定到 mm_inputs。直接套用会提前释放仍被模型使用的存储。 |
| `b822c05fdf33ed6eb8f88d572f776f05b218af44` | fix(disaggregation): stop unsent encoder parts after abort | 已覆盖 | 目标现有实现 | 目标 `encoder/receiver.py` 的 abort 通知、`encoder/transfer_queue.py` 的 abort registry/队列清退及 `test_encoder_abort.py` 已处理未发送任务取消和运行任务排空；保留目标实现。 |
| `71a95a45af9b1240ed5c42094c8014ba85aa8356` | feat(observability): export host mamba pool used/total slot gauges | 迁移 | 2cec40bb28 | 指标写入 `observability/metrics_collector.py`；采样位置改为 `managers/scheduler_components/metrics_reporter.py:_log_hicache_stats`，缺少 Mamba host pool 时清零。 |
| `ce99439c2e8d9e23003efd8dd96a120ae631b382` | fix(disagg): let stub-bootstrap benchmark requests enter the radix cache | 迁移 | abc104c533 | 保留目标 rebootstrap 字段及 pinned-array 导入，在 `Req` 中增加默认关闭的 fake-bootstrap radix opt-in。 |
| `4b98a3bba25d622a2fe1ebfb035f07e1246bc69f` | MTP 复用 NSA DSA init_forward_metadata | 跳过 | — | 源 NSA/Mamba metadata 的共享路径不存在。目标 GLM 使用 DSA backend 与 in/out-graph metadata，Mamba 槽位有目标池翻译；不跨 target/draft 池直接共享源 metadata 对象。 |
| `f03b2ad3574e86a44174fd590abf303b961c3c85` | feat(kda): add env-gated TileLang target_verify kernel (odd-S capable) | 迁移 | 5f9525641d | 与后续安全修复合并移植；目标入口为 `linear/kernels/kda_hcu.py:HcuKDAKernel.target_verify`，kernel 位于 `kernels/ops/attention/fla/hcu/`。 |
| `3d08bb73b8df5025f6bed3ec7731ce1aee755d66` | fix(kda): review fixes + stride/fp16 perf for the TileLang verify kernel | 迁移 | 5f9525641d | 保留 padding/stride/fp16/fallback 修复和预编译；目标开关为 pool 的 `use_hcu_kda_replayssm`，capture API 适配见 `de2b688e33`。 |
| `8fcefa0916bcb9ba1db61db5352ba94e13b8b86a` | perf(encoder): dedup the EPD decode-side multimodal preprocess | 跳过 | — | 目标没有源 content-key encoder cache/probe、`mm_content_key.py` 与对应批处理协议；目标 encoder 已拆为 runtime/preprocessor/server，保留当前协议。 |
| `2387a27403f2aff1834266b35bad0c8374f90af6` | fix(mm): port upstream canonicalize/rendezvous + content-hit guard + scheduler seed | 跳过 | — | 依赖上一条缺失的 content lookup，包括 URL canonicalize、rendezvous/content-hit grid guard；不引入孤立开关或旧 receiver/server。 |
| `2dc443654559859c9f547d27f33f654d90a82692` | feat: support w4a8 deepgemm. | 已覆盖/保留差异 | 目标 241207e321 | 目标已有 SlimQuant W4A8 DeepEP/DeepGEMM 与 HCU compressed-tensors W4A8；保留其 checkpoint 格式与 scheme 分派。源独有 `w4a8_int` compact-config 别名及旧 MoE scheme 不覆盖目标。 |
| `7b361c36b84d1c43aad4455adf2ff61d35ee3a88` | Supports both startup parameters '--enable-kda-replayssm-spec' and '--mamba-ssm-dtype float16'. | 迁移 | 072def91a7 | 参数 hook 已允许低精度并告警；移除 HCU commit kernel 和 MambaPool 中剩余 fp32-only guard，保留 FP32 g/beta 和累加。 |
| `155d946de36506c568f8feb424779674a2990409` | fix w4a8 kernel fusion issue. | 迁移 | 9fb5493552 | 按具体 scheme 公布 prequantized-input 能力，消费已有 INT8/scale；保留目标 LM-head/KV scheme 与签名过滤。通用 factory 修正见 `de2b688e33`。 |
| `a19c0ab7d760e16246150be1d78ddffc2e01496a` | perf(pd-decode): remove host syncs that stall kernel launch on batch-size change | 部分迁移 | 9ab9cc0cc9 | 移植 PREBUILT 小张量 pinned/async H2D 与 KVALLOC 默认关闭，并适配微基准路径。目标 sampling 已异步；源 `new_seq_lens_cpu`/SeqLensCpuView filter/merge 流程已被 relay 管线替代，跳过该部分。 |
| `7e29940662c6078483c6287a354ad7633468c297` | fix(pd-decode): keep a full MTP seed for fake-prefill requests | 跳过 | — | 目标 Req/FutureMap 不使用源 `mtp_topk_indices_tensor`/`mtp_topk_indices_dim` PREBUILT seed 协议，而由 spec algorithm/relay 构造输入；不添加失效字段。 |
| `8c2af29cb973c8d0a06da48f895a95a91e2d880b` | bench: interleave shared-prefix requests and validate warmup | 部分迁移 | 36933d9a4e | 默认 uniform 新数据 round-robin，warmup 取首批真实请求分波发送；保留目标 ordered/Zipf、warmup 剔除及至少一条正式请求的边界。源 auto_benchmark.py/lib 不存在，跳过其 output-length 工具改动。旧数据缓存保留原顺序。 |
| `b41bbf6680c6bf6f423c38d0d40ae1aecf1f8258` | fix(encoder): avoid decoding video pixels for metadata | 跳过 | — | 目标不存在源 metadata-only video decode endpoint/调用链；当前 `encoder/preprocessor.py` 走自己的媒体预处理与 shard 元数据流程。 |
| `b7c10b7f1aea5940eee79d29c85edc4925754694` | fix(encoder): bound CPU video decode buffers | 跳过 | — | 目标没有源 `_decode_indices_parallel`/`sample_video_sglang_sync` CPU bucket decoder，不能把源整段 GLM processor 覆盖回来。 |
| `011f5e65aa53ee8882161b04024de8dde50e547e` | fix(encoder): pin layout and tensor format for video metadata | 跳过 | — | 仅修复源 metadata-only meta/TCHW 输入，依赖 b41bbf6680 的缺失路径；目标没有该 meta 调用。 |
| `74c8e36b5e03e686c5bec431a16b972a7b40d533` | feat: add opt-in CP sharding and prefetch for MLA weights | 跳过 | — | 目标没有源 `zero3_attn_weight.py` / `distributed/attention_cp.py` 的 replicated-attention-weight sharding/prefetch 功能；保留目标 CP/GLM DSA 实现，不导入整套新权重生命周期。 |
| `2e249dcecd7835ee447fb538edc89c2fcf22de66` | perf: optimize CP attention weight prefetch with HIP IPC | 跳过 | — | 依赖上一条缺失的 ZeRO-3 attention-weight prefetch，HIP IPC 权重读取及其更新保护没有目标消费者。 |
| `edec1245bdac657da9c93db0b50485d872d7d6dc` | fix(disagg): close stale ZMQ sockets to a failed prefill on node failure | 部分迁移 | d1b554a679 | 目标 common manager 已统一 Mooncake/NIXL node-failure 并清理 receiver sockets；补充 manager socket/monitor 清理及 send-lock 排空。源 lease-based `ZMQSocketCache` 不存在，保留目标字典缓存。 |
| `bf9db3629eccee8e1c2e41099c49f21ac89ac606` | fix(glm4v): require begin token before mm placeholders in EPD prompt rebuild | 已覆盖 | 目标现有实现 | `multimodal/processors/glm4v.py:build_input_ids_with_timestamps` 已要求完整 begin + modality token 锚点；保留无 begin-ID 时的兼容分支。 |
| `aa1732a1e998e5754faef40c2ffe5b3b150c69ba` | update deepgemm. | 已覆盖 | 目标 4c095c2130 | 目标已有 `m_grouped_i8_gemm_nt_masked` 与 `marlin_i8_masked_weight`，对应工具改名为 `hcu_deepgemm_w8a8_utils.py`。 |
| `d3db5682d1560c4d707feb4d17a92edac1b3c91f` | fix(epd): bound encoder host memory across request lifetimes | 跳过 | — | 针对源旧 encoder HTTP body admission、content-hit-before-reserve、cache storage accounting 与 `/send` 生命周期；目标没有这套 admission/cache 协议。保留目标 scheduler/runtime/transfer_queue，不声称已具备源全部 host-memory 限额保护。 |

