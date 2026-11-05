# R02：beta-only增量参考补测

R01候选两轮4807.52／4807.13ms，未复现F组4672.91／4673.19ms；前后关闭对照均约5018–5020ms。固定配置／模型metadata／native及通信库哈希与筛选一致，已记录的相关初始环境也相同。F组beta-only为4786.69ms，因此需要一个同条件重复参考，才能判断新增shared＋CP overlap＋Q直写的增量，不能把总收益全部归于新框架。

本补测为**单独一组beta-only参考**，内部目录`R02-candidate`复用现有安全执行器的命名规则；R02不声明为另一轮关闭A／候选／关闭B。

- 同一最终15文件源码／LightOp B2／原FlashMLA与DeepGEMM；从已完成F-beta复制启动脚本，字节哈希留档。beta=1，所有9个新增开关为0。
- 固定112640输入、107008缓存、输出1、BS8、32请求、chunk32768、TP8/CP8/EP8、CPv2 interleave、EAGLE5/1/6。先跑既有5-shot20题sanity，再按原种子2026110401／02跑两轮无profiler正式性能。
- 正式性能结束后冷flush，复用既有20-shot100题、greedy／parallel4／max512评测；保留原解析器、accuracy>0.8及全部分数／逐题结果，记录实际生成JSON，与R01按题号核对语义哈希。前20题属于示例，另报非示例80题。
- 引用已完成F-beta的同源码／包／开关8rank trace，明确本补测无新profiler。原始输出、命令、manifest、8rank源码／native／通信库／环境及HCA验证保存在独立目录。
- 启动前置条件：R01三个角色全部归档、精确PID清理和8卡释放核验完成；本地`R01_results.json.complete=true`；另做本轮空卡／无KFD PID／端口及源码体检。单独补测结束后同样精确清理、核验释放。

**本组已完整结束并核验8卡释放。** 两轮4793.04／4799.03ms、均值4796.03ms；长100题96分、非示例76/80。完整证据见[R02.md](R02.md)。 完成后与R01候选及前后关闭对照分别比较；若新增开关没有稳定增量，则保留关闭并如实记录，不由旧筛选均值推定有效。

CP/Q组合复测未显示相对本组的额外收益；原矩阵另两组候选的独立确认协议见[FRAMEWORK_CONFIRMATION.md](FRAMEWORK_CONFIRMATION.md)，本组作为其前置beta-only参考。
