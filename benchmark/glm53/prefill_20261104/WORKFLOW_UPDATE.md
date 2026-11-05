# 2026-11-05 最新交付顺序

按用户最新要求：已有相对base收益和正确性证据后先提交PR，多个优化点可以一并在分支中独立开关；兼容点的最终组合测试在PR后继续补充。相对beta-only的独立增量不再作为提PR前置条件。所有测量、未复现收益和精度限制仍如实报告。

所有优化代码、单项记录和后续组合结果统一提交到[feat/glm53-prefill-opt](https://github.com/maxiaobetter/sglang-das/tree/feat/glm53-prefill-opt)。到旧分支`fix/dsa-hicache-index-k-layer-mapping`的[汇总PR #52](https://github.com/maxiaobetter/sglang-das/pull/52)现为Draft，待组合验证完成后再转正式审核。未拆分优化分支，也未合入旧分支。

发布头`712c959501d4a85f72951d22446f3dae48a244ba`的完整PR diff已通过项目CI等价pre-commit、15个Python文件语法、4份命令快照bash语法及diff检查；GitHub的lint和构建任务仍排队，尚不能记为CI通过。后续提交继续遵守相同lint要求。

已完成单项和原矩阵筛选，以及R01整体低4.22%、R02 beta-only低4.44%及100题检查。初测6.88%未复现，新CP/Q组合相对beta-only没有额外收益证据；这些结论不改写。

原先排队的R04 shared-only／R05 beta-only增量比较未启动，已取消该旧队列；R03 KV＋shared在运行，保留已有模型和客户端，由单组收尾控制器负责归档／精确PID清理／8卡释放。后续按兼容关系组合有正向信号的点：P03与P10互斥，P10已含Q拼接；P06回退、P09精度疑点及不适配Mega保持关闭。组合运行和结果更新不阻塞本次PR创建，也不被提前标为完成。

最终整个目标仍包括PR／相关CI和后续组合结果；本次调整改变交付顺序，不放宽原数值／评测阈值，不新增仓库单测，不本地跑完整套件。
