# 交付核验

**当前整个目标未完成。** 全部优化已统一推送到`feat/glm53-prefill-opt`，[汇总PR #52](https://github.com/maxiaobetter/sglang-das/pull/52)保持Draft；后续组合验证和相关CI仍待完成。

| 原要求 | 已检查的证据和范围 | 当前结论 |
| --- | --- | --- |
| 1．实现与运行／正确性／性能分开 | 15个Python文件与实际部署manifest逐字节相同，代码快照7684；PRE_PR_SOURCE_CHECK.json及IMPLEMENTATION_REVIEW.md。P03/P04/P05/P06分别有独立记录和profile，P07临时合并Mega adapter移除、原后端保留 | 接入、命中和成绩分别记录；未把实现等同收益 |
| 2．审计全部适用方向 | COVERAGE.md、P03至P13各项文档。Q投影与Q absorb overlap分开；P10处理CP全局loc／Q stride；P05/P08/P09有显式ABI及原生比较；B0已有融合／复用未重记；LayerSplit限PD，Mega当前pin的量化／内存／fabric边界有源码和实测依据 | 已审计／分类；P09低分及原生差异保留且关闭，P06性能回退关闭；当前包不适用项不称已适配 |
| 3．保留B0与历史 | BASELINE_PRESERVATION.json：baseline.json和四份命令快照字节等于最初提交，B0/P01/P02账本原始字段等于首次账本；baseline仍5011.77ms，原beta4802.42ms、候选Flash4973.28ms独立保留 | 已核对引用／字段；没有把新包或开关并入B0 |
| 4．独立及组合验证 | 已进入模型性能矩阵的候选有同配置／种子、两轮无profiler与独立主32K trace，换包另有B2关闭对照；源码／实际native／通信库／环境／HCA及8rank命中证据留档。R01六轮及三组100题实际生成JSON语义SHA相同；95/96/96分，非示例75/76/76 | R01整体低4.22%，筛选6.88%未复现。R02 beta-only4796.03ms／96分已完成；CP/Q组合额外收益未见，剩余KV／shared候选的重复确认仍在进行 |
| 5．真实来源、ABI和资源 | AUDIT.md／LIGHTOP.md／P13_source_audit.json及各case runtime。三库实际SO按rank核对、隔离B2与原包分开；Torch／DTK／CXX11／gfx938记录。当前模型indexer32头／128维与16例既有MQA用例一致。R01三组精确PID／启动时间清理、另核8卡2MiB | 来源／部署身份及R01历史释放已检查；R02已归档并核验8卡释放；当前剩余候选确认逐组体检／清理 |
| 6．最终交付门槛 | 发布头712c959501的完整PR diff lint、15个Python语法、4份命令快照bash语法、diff检查及部署源码对齐均通过。未新增仓库单测、未本地跑完整套件；同一优化分支及Draft PR #52已核实 | **未完成：相关GitHub CI、后续组合更新及Draft转正式审核** |

所有正式压测逐请求核对112640输入、107008缓存、输出1、32/32成功。100题使用原解析器／accuracy>0.8；前20题属于示例，另报非示例80题。原生MQA保留mask等值／calc_diff<1e-3，P06原速度断言失败仍保留。score、局部数值用例、编译与lint各自说明范围，未互相替代。

模型config／tokenizer／index metadata哈希留档，同一755.86GB权重目录未采完整权重内容SHA。主32K step按rank0报告，跨stream累计／交集不当因果节省，未采HBM PMC；其他模型、拓扑、Graph、PD和完整准确率套件不属于已验证范围。

R01_results.json为已完成且冻结的六轮／逐题证据；R02为单独补充参考，不能写成另一轮A/B/A。完成剩余候选确认后需更新本表及DELIVERY.md，只有所有必需门槛都有实际证据，整个目标才可标为完成。

交付顺序以[WORKFLOW_UPDATE.md](WORKFLOW_UPDATE.md)为准；不再把相对beta-only的独立增量或旧R04／R05队列作为PR前置条件。
