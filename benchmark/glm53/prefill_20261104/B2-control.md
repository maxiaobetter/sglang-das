# LightOp B2：全部优化关闭的匹配对照

仅使用隔离 LightOp B2，九个新优化开关全部为0，`LIGHTOP_MQA_LOGITS_BETA` 未设置；原 FlashMLA、DeepGEMM、DeepEP、RCCL 沿用。框架源码固定 `69fbccd6ed`，不能将新包和融合同时变化后归因于某一融合。

| 项 | 结果 |
| --- | --- |
| 两轮无profiler TTFT | 5017.95 / 5016.95 ms |
| 均值 | 5017.45 ms |
| 相对历史B0 | +0.113% |
| 8卡新输入吞吐 | 8717.55 tok/s |
| 既有GSM8K前20题 | 19/20，invalid0，原阈值>0.8 |
| 主模型32K chunk | 3272.20 ms |

换包关闭状态基本对齐原对照，不能记作端到端加速。P05/P06/P08/P09/P10 分别与这组比较，历史B0独立保留。每轮32/32成功，逐请求输入112640、缓存107008、输出1；种子和原正式压测脚本不变。

两台各4个rank：16个源码文件及三个目标native库哈希通过；附带LightOp辅助库均在包manifest允许集合内。已从各自存活Python launcher采集完整初始环境（敏感字段脱敏），逐一确认九个开关为0、beta未设、精确PYTHONPATH；8rank的DeepEP与RCCL哈希一致。每台5个HCA端口ACTIVE。八份trace gzip和hash已核验。小样本GSM8K只作本次回归sanity，不代表完整精度。

模型运行读取隔离前缀 `lightop_c3881932743e_prefix`，未覆盖site-packages。wheel SHA256 `bcf4b829fb6d003da60fa8627b82f30d4e68b4c9f9fea221a082ece0eed75f41`；LightOp op.so SHA256 `f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c`。

证据远端 `/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515/B2-control`，本地 `/Users/better/Desktop/SGLang相关/sglang_dcu_github_origin/dcu-sglang/profiles/glm53-prefill-opt-20261104/B2-control`。rank0 trace SHA256 `c9a3d0081ec46d1f1762ed5b6fc648c480c2d0d459b441ed023c00c5ef760597`；八份原始trace远端保留，rank0已下载分析。服务按本轮PID及启动时间退出，另行确认8卡均2MiB后才启动P05。结构化记录见 `B2_control_results.json`。
