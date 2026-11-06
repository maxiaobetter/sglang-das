# GLM R09 prefill 配置

本分支只提取 R09 的 KV 融合、packed CP 交换和共享专家 overlap，并提供 LightOp beta 环境配置。两个新增框架开关默认关闭。**不包含 MegaMoE**、Q 投影重叠、其他 CP overlap、Q 直写或额外量化实验。

在已有服务环境之后 `source env.sh`，不需要增加 CLI 参数。已验证的配置为 channel-FP8、DeepEP NORMAL、TP8/CP8/EP8、interleave CP、chunk32768、eager。KV 融合仅适用于普通 prefix prefill、BF16 激活/RoPE、656 字节 FP8 缓存；不满足守卫时保持原路径。共享专家 overlap 仅用于普通 GLM prefill、独立共享专家且没有 SBO/TBO 或 graph capture 的情况，不修改专家权重量化。

配套 LightOp B2 源码为 `c3881932743ed63e2fd6c30c8474d46ac8d5c15a`，环境为 DTK26041、Python3.10、gfx938，必须支持 Q head stride 256。保留配套 FlashMLA/DeepGEMM/DeepEP；成功导入不等于实际运行验证。

## 本次目标分支验证（2026-11-06）

- 基线：`b609db3174b2f9f79ded1878488355a07d5e5fdd`；修复后代码：`192cd614172a7c7acaff178ee151cc04e3a00e04`。
- p2c2+p2c3，容器 `sglang_glm_mx`，共 8 rank。基线与候选均开启 beta，仅候选开启两个框架开关；源码使用隔离前缀，安装目录未替换。
- 实际加载的 LightOp `.so` SHA256：`f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c`。每 rank 的库路径、哈希、模型元数据和启动环境均已保存；pip 分发元数据不能代替实际加载文件。
- KV 同实际输入对拍：8 rank × layer0/3/77，共 24 组；Norm、Q/K RoPE 最大差值均为 0，Q 非旋转部分一致，FP8 打包字节及 CP 交换后的全局缓存字节一致。
- 共享专家同输入对拍：8 rank × layer3/77，共 16 组，每 rank 4096 tokens。overlap 开/关确实使用不同 stream，共享专家输出与 MoE 最终输出最大差值均为 0。
- 固定 GSM20：基线与修复候选均为 19/20，invalid=0。它是有限的回归检查，不是完整模型精度验收。
- 两组长前缀均为 32/32 请求成功，输入112640/缓存107008/输出1，全部长度逐项匹配；两组各有 8 rank 长前缀和 8 rank MTP profile。固定 1024 输入、64 输出的 MTP 请求文本完全一致。
- 全部 32 份 trace 已解析。候选长前缀每 rank 命中 936 次融合 `rms_rope_kernel`，KV 融合标记均限于普通 EXTEND；MTP profile 各 rank 覆盖 decode、verify、draft extend，未命中 KV 融合。两组实际内核均为 DeepEP＋DeepGEMM，无 MegaMoE。
- 服务按 PID 与启动时间核验后清理；p2c2/p2c3 共 8 卡均已释放到每卡 2 MiB。

首轮验证发现提取代码将 packed Q 整体传入，而目标 DSA prefill backend 要求独立的 `q_rope`，触发断言。修复在解包时将 Q 切为前 512 维、后 64 维两个 view，保留 `save_kv_cache=False`。未移除后端断言、未修改数学或放宽数值阈值。首轮失败证据另行保留。

对拍 hook 只放在独立诊断副本中，没有提交到框架；诊断运行不作为正式性能测量。沿用现有数值门槛，没有新增仓库单测。已运行项目 CI 同配置 lint、Python 语法、`bash -n` 和 `git diff --check`。

证据根目录：`/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515/`，分别为 `PR54-base-validation-p2c23`、首轮失败的 `PR54-candidate-validation-p2c23`、修复后的 `PR54-candidate-r2-validation-p2c23`。其中 `pairs/`、`*_runtime.json`、profile manifest 和日志记录实际证据。

**边界：本轮不验证 W4A8 模型、PD、其他并行布局或 CUDA graph 重放，也不以历史 R09 性能替代本提交的性能验收。** 在不同环境开启时，仍需核对实际算子包/ABI、各服务 rank 的调用链、shape/stride/dtype、量化与缓存布局；新增路径需补齐相应验证。

## 历史性能，仅作参考

原 R09 冻结环境使用 channel-FP8，输入112640/缓存107008/输出1，BS8、32请求、32K chunk，p2c1+p2c9：

| 配置 | 平均 TTFT（ms） | GSM20 | 100 题 |
| --- | ---: | --- | --- |
| beta＋KV融合＋共享专家重叠 | 4699.16 | 19/20 | 96/100 |
| 结束 beta 对照 | 4787.75 | 19/20 | 97/100 |
| R09＋Q投影重叠（未提取） | 5031.86 | 19/20 | 96/100 |

R09 相对结束 beta 对照 TTFT 低约 1.85%。这些数据来自旧部署，不能外推到本提交、W4A8、PD 或其他 BS。

原始提取来源：FP8 快照 `7684f58399ad70de47de7678897f7ea5aab6107e`、KV 提交 `ca56102ac6`、共享专家 overlap 提交 `c3343455c0`。提取保留目标分支 `_forward_shared_experts(hidden_states)` 接口与原专家计算，仅调整 stream、事件和张量生命周期。
