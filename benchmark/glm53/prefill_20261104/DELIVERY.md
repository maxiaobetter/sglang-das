# GLM5.3 普通 prefill 优化交付记录

**当前状态：筛选及R01交替复测已完成；R01整体低4.22%，未复现筛选6.88%。R02 beta-only已完成，均值4796.03ms、长100题96分；剩余KV／shared候选独立确认仍在进行，尚未提PR。** R01阶段结果见[R01.md](R01.md)，完整逐项证据见[README.md](README.md)、[COVERAGE.md](COVERAGE.md)。

## 固定复现条件

- p2c1＋p2c9，容器`sglang_glm_mx`，8张gfx938 DCU，TP8/CP8/EP8、CPv2 interleave，普通IFB，LayerSplit关闭、eager。
- 输入112640、共享缓存107008（95%）、新输入5632、输出1；全局BS8，每轮32请求，chunk32768，EAGLE5/1/6。每轮逐请求核对输入／缓存／输出，正式性能不开profiler。
- 部署为冻结的原安装源码副本加15个本分支Python文件，代码快照`7684f58399ad70de47de7678897f7ea5aab6107e`；后续文档提交不改变这15文件的字节。分支基点不被当作容器完整源码身份，完整来源和哈希见各实验manifest。
- 固定LightOp B2、原FlashMLA、原DeepGEMM；没有GEMM选优／skinny改动，没有EPLB映射。模型metadata哈希保留；同一755.86GB权重目录未采完整权重内容SHA。
- 原B0仍为5011.77ms／8724.79新输入tok/s；旧原包beta-only 4802.42ms、候选FlashMLA 4973.28ms分别保留，未并入B0。

## 同包同源码筛选

| 配置 | TTFT均值 ms | 相对关闭 | rank0主32K step ms | GSM20 |
| --- | ---: | ---: | ---: | --- |
| 同包同源码关闭 | 5018.46 | +0.00% | 3276.91 | 19/20 |
| 仅MQA beta | 4786.69 | -4.62% | 3116.19 | 19/20 |
| beta＋shared overlap | 4733.11 | -5.69% | 3079.72 | 19/20 |
| beta＋shared＋CP overlap＋Q直写 | 4673.05 | -6.88% | 3026.76 | 19/20 |
| beta＋shared＋KV融合/packed交换 | 4701.12 | -6.32% | 3053.06 | 19/20 |

每行两轮正式测量，独立profile；客户端BS8与主step中的bs6分别报告。筛选较快组合已按预定规则选入R01，不能只因均值最低就宣称稳定收益。beta沿用已有LightOp入口；新框架的增量与beta单独结果分开记录。32K step及跨stream kernel累计量不能直接换算成整请求TTFT贡献。

## 选入复测的配置

```bash
export LIGHTOP_MQA_LOGITS_BETA=1
export SGLANG_GLM_PREFILL_CP_OVERLAP=1
export SGLANG_GLM_PREFILL_DIRECT_Q=1
export SGLANG_GLM_PREFILL_SHARED_OVERLAP=1
export SGLANG_GLM_PREFILL_NORM_QUANT=0
export SGLANG_GLM_PREFILL_NATIVE_KV=0
export SGLANG_GLM_PREFILL_SHARED_QUANT=0
export SGLANG_GLM_PREFILL_INPUT_NORM_QUANT=0
export SGLANG_GLM_PREFILL_KV_ROPE_STORE=0
export SGLANG_GLM_PREFILL_Q_PROJ_OVERLAP=0
```

所有新增开关默认关闭。选中组合实际确认78次主KV AllGather复用同一独立通信stream，完整Q拼接消失、改为78次RoPE尾部copy；shared独立stream与routed计算重叠。区间交集不等于因果节省，完整记录见`F-beta-comm-copy-shared_results.json`。P10与P03互斥，P10已包含Q拼接。

## 各方向结论和边界

- P03-r2／P04独立测量变化小；其与P11、beta的组合进入R01，尚待重复确认。P11独立TTFT低1.58%，P12 Q投影overlap仅低0.52%，不将两种overlap混为一项。
- P10 KV norm/RoPE/store及656-byte packed交换独立低0.77%，本次组合慢于选定CP/Q方案，保留关闭。payload每行1152→656byte，仅指新KV交换；不声称整体网络流量或实测HBM带宽同比下降。
- P05 Q norm/quant、P08 shared SwiGLU/quant未确立端到端收益，保留关闭。P06数值通过而原速度断言失败，模型TTFT高2.18%，保留关闭，原始失败保留。
- P09 input/residual norm/quant存在未定位的sanity下降：17/20、冷复核17/20，对照19/20；更多原生输入有少量差异，保留关闭、不进组合。旧样本逐位一致不泛化到任意输入。
- 原有indexer norm/RoPE、QK量化/写缓存、TopK复用和小窗口通信重叠已在B0命中。下一层KV预取的LayerSplit限PD prefill，本次普通IFB未删除守卫来测试。
- P07合并Mega临时adapter已移除，原有后端保留。P13独立Mega normal支持GLM shape，但当前K2舍入／scale下限／路由权重位置与固定GLM管线不等价；raw＋packed权重生命周期超预算，当前binary的fabric格式为64B回退。用户补充仓库main@91cdff7已核对，关键FP8/fabric附带源码一致，后续更新集中W4A8。GLM兼容K2、权重生命周期与HSA构建仍未实现，不计性能收益；见[P13.md](P13.md)。

## LightOp包和正确性证据

固定B2源提交`c3881932743ed63e2fd6c30c8474d46ac8d5c15a`，Py3.10／DTK26041／gfx938／CXX11 ABI=true；Torch commit为`a346d24b0aa21d34ebda15ba9839433496b9d887`。包与源码归档／patch的完整位置及哈希见[LIGHTOP.md](LIGHTOP.md)、`LightOp_source_artifacts.json`。

```text
/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515/lightop_c3881932743e/dist_candidate/lightop-0.6.0+das.dtk26041-cp310-cp310-linux_x86_64.whl
```

wheel SHA256：`bcf4b829fb6d003da60fa8627b82f30d4e68b4c9f9fea221a082ece0eed75f41`。
op.so SHA256：`f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c`。
本地wheel、源码归档和patch位于原工作区`profiles/glm53-prefill-opt-20261104/lightop-build-b2/wheels/`；原脏LightOp目录未混入交付。

沿用既有数值断言：MQA FP8普通/CP mask共16例、H32/D128（与实际indexer一致），原calc_diff<1e-3不变；KV RoPE/store复用168＋2个GLM stride256用例。P06原速度失败和P09更广输入差异分别保留，不把编译、lint或局部数值用例当作全模型准确率保证。

R01按关闭A→候选→关闭B，每组两轮无profiler正式性能及既有20-shot／100题、greedy／parallel4／max512评测；原解析器和accuracy>0.8不变。前20题同时在示例中，另报非示例80题；记录实际生成JSON、全部回答及元数据，逐题核对语义SHA。历史F-control97/100、beta-only96/100；R01关闭A95/100，关闭组也新增错27/33，当前不能由旧1题差异唯一归因于beta。R01候选现为96/100（非示例76/80），两轮TTFT4807.52／4807.13ms；相对前置关闭A低4.23%，未复现筛选6.88%。两组实际100个生成JSON语义SHA相同，候选改对27/33、新增错87；关闭B为5018.35ms、96/100（非示例76/80）；R01前后关闭四轮均值5018.96ms，候选低4.22%，三组实际100个生成JSON语义SHA相同。完整原结果保留，R02 beta-only为4796.03ms、96/100（非示例76/80），CP/Q组合比它高0.24%，未显示额外增量；KV／shared另两组的重复结论待确认。

每组按确切PID／启动时间清理，另核8卡释放；运行记录为历史资源状态，不代表这些节点现在可用。原始证据根目录：

```text
/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515/
```

## 尚未完成的交付门槛

原矩阵另两组KV／shared候选的独立确认及新增框架增量判断、最终推荐；最终源码15文件与PR内容再次核对；按项目当前CI配置跑最终diff lint、Python语法／脚本语法及diff检查；向`fix/dsa-hicache-index-k-layer-mapping`提PR并处理相关GitHub CI。其他模型、拓扑、Graph、PD和完整准确率套件不在本次已验证范围，不声称普遍加速。

针对复测4.23%与筛选6.88%的差异，另准备同包同源码的单组beta-only参考，协议见[BETA_REFERENCE.md](BETA_REFERENCE.md)。R02已完整结束，成绩见[R02.md](R02.md)；CP/Q组合的额外增量未复现，另两条原候选的确认见[FRAMEWORK_CONFIRMATION.md](FRAMEWORK_CONFIRMATION.md)。

用户最新要求为先提交已有base收益及正确性证据的PR，兼容点组合后补；当前在途R03保留，旧增量队列取消。交付顺序见[WORKFLOW_UPDATE.md](WORKFLOW_UPDATE.md)。新增开关默认关闭、收益及精度限制继续分别记录。
