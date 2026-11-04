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
| P03 主 KV AllGather / Q absorb BMM overlap | CPv2 interleave 已接入，保持 logical/physical padding；配置只启用此新增开关 | r1 已完成两轮与profile，明显慢于历史B0；fresh control重现B0，r1回退确认；r2恢复基线附近、TTFT低0.66%尚不宣称稳定收益，详见P03.md |
| P04 直写 Q buffer | Q concat已由78次full copy改为tail copy；BMM strided输出略慢 | 两轮TTFT低0.53%，暂不宣称稳定收益，详见P04.md |
| P05 Q RMSNorm + FP8 quant | 基线有 `rms_norm_fp8_quant`，但新接入要求显式 `match_sglang` ABI v1，避免沿用旧归约/量化行为 | GSM8K19/20、78次融合命中；TTFT高0.15%而32K step基本不变，默认关闭，见P05.md |
| P06 native KV gather/dequant | 基线有 kernel 符号，但缺少 `PREFILL_GATHER_TRITON_NAN_COMPAT_VERSION`，不能只凭符号开启 | B2数值通过、原速度断言失败；模型GSM19/20及8rank命中，TTFT高2.18%、32K step高88.07ms，默认关闭，见P06.md |
| residual/input norm + quant | 基线无显式 `match_sglang` ABI v1；兼容内核将以 opt-in 保持 FP32 residual 语义，默认 API 保留原行为 | 融合与消费者命中，TTFT低0.09%无稳定收益；GSM17/20复核17/20，不进组合，见P09.md |
| P10 KV norm/RoPE/store 融合 | 新增prefix CPv2 interleave入口，先写local656-byte行，再全局byte gather及physical loc store；不重复量化，不改cold BF16路径；限制普通DSA pool | B2原生168+2组通过；GSM19/20、8rank命中，TTFT低0.77%、32K step低15.37ms，默认关闭待组合，见P10.md |
| 合并版MegaMoE | 原型曾对应develop的deepgemm.mega，但原生L1/L2实际使用groupwise激活scale | 固定GLM per-token约束下排除；新增adapter已移除，现有后端保留；另有双宿主HIP IPC失败，见P07.md |
| 历史 KV 下一层预取 | 分支已有 `maybe_prefetch_next_full_attention_kv`，依赖 layer split；B0 的 IFB layer split 关闭 | 普通IFB不允许启用LayerSplit，server_args限定PD prefill；本协议不直接加flag，详见COVERAGE.md |
| P11 shared expert overlap | `forward_deepep` 已有 alt_stream + event；B0未开 `SGLANG_ROCM_USE_MULTI_STREAM`，MoE将alt_stream置None。P11复用已有组织，只对GLM普通prefill启用一条缓存stream，显式记录跨stream生命周期 | 单项两轮TTFT低1.58%，GSM8K19/20，shared stream命中；稳定收益尚未确立，见P11.md |
| P12 Q projection / indexer overlap | 原 `q_b_proj` / indexer重叠只在capture decode门控内；新增独立prefill开关，只在indexer确实执行时启动非collective投影并join | GSM8K19/20、并行命中，TTFT低0.52%但无稳定收益，默认关闭，见P12.md |
| P08 shared expert SwiGLU + quant | 旧融合核带最小scale，基线per-token量化不带；已改为要求显式SGLang SwiGLU FP8 ABI v1，保持BF16激活/乘法中间舍入，再按amax直接量化 | B2原生8组逐位一致；GSM19/20、75次融合/消费者命中，TTFT低0.071%无稳定收益，默认关闭，见P08.md |
| indexer overlap / TopK reuse | 基线21个full/57个shared层，已有21次Indexer-K AG与query Hadamard重叠；FP8量化/store融合也已命中 | 覆盖核验已完成，详见COVERAGE.md；不计新增收益 |

P03 与 Q projection 的 overlap 是不同候选，不合并记账。P01 MQA beta 和 P02 FlashMLA 的已有实测保持独立，见 README 和 results.csv。
任何仅完成静态检查、启动加载或符号探测的项目，均不能填写“正确性通过”或“性能提升”。

## LightOp 兼容候选

独立 worktree `lightop-glm53-prefill`、分支 `feat/glm53-prefill-fusions`、提交 `8c2b40a8`，基于已核对的 master `cf5b95407474`。P05/P09 要求显式 `match_sglang=True`，原 API 默认行为保留；额外声明 GLM Q head stride 256。B1 wheel已编译完成且ZIP完整性通过；后续 c3881932743e 增加P08显式兼容入口的B2已编译，native数值检查通过，P06性能断言失败；同包关闭对照及P05/P08/P09/P10已独立实测；P09有未定位sanity下降、P05/P08无稳定收益、P10小幅变化待组合确认，不能将包能力当成模型收益。

对4个变更文件，已运行仓库 quality-gate v2.0.3 的实际 code-quality/governance 策略与解析器；Ruff 0.12.1 与 CI 镜像一致，使用本地工具代替 Docker，Lizard 1.25.0 为 advisory。两组均0 finding，Python语法及 diff check通过；security gate 和 native build 不在此 lint 结论内。完整报告保存在原工作区 `profiles/glm53-prefill-opt-20261104/lightop-lint/`。

独立 `megamoe` 已单独完成当前包适用性核验：normal支持GLM shape，但实际8组K2量化对照不同，raw+packed最低168.75GiB超设备预算，native fabric返回64B；当前包不作为固定GLM量化管线的原样替换。未做完整模型性能，不将P07失败直接套用；未实现兼容ABI/生命周期/HSA编译等见P13.md。
