# 最终源码对照与组合矩阵

以下均为独立实验ID；尚未取得终态结果的行不记为通过或收益。使用最终代码快照 `7684f58399ad70de47de7678897f7ea5aab6107e`，15个改动Python文件与后续仅文档提交的代码内容相同；相对单项使用的69快照，只移除不适配的合并Mega原型、恢复原有Mega入口，其余优化代码不变。隔离LightOp B2 f4d69b、原FlashMLA b0ea2f、原DeepGEMM79325b。B0、旧framework-control、B2-control及所有单项结果保持独立。

固定112640输入／107008缓存／1输出、BS8、每轮32请求、chunk32768、TP8/CP8/EP8、EAGLE5/1/6；正式种子2026110401/02，独立profile2026110403。下表未列的新增开关全部0，P09不进组合，P06已确认模型回退，不加入组合。

| ID | beta | 新增开关 | 配对目的 | 初始状态 |
| --- | --- | --- | --- | --- |
| F-control | 未设置 | 全部关闭 | 最终代码＋B2关闭对照 | 单配置测量及长100题完成，见FINAL_RESULTS.md |
| F-beta | 1 | 全部关闭 | 相对F-control只变MQA beta；替代旧69源码上重复的P01-B2跑测 | 已准备、未验收 |
| F-beta-shared | 1 | P11 shared overlap | 相对F-beta只变shared overlap | 已准备、未验收 |
| F-beta-comm-copy-shared | 1 | P11＋P03-r2 CP overlap＋P04 Q直写 | 相对F-beta-shared测两个已独立测量项的组合交互，不作单项归因 | 已准备、未验收 |
| F-beta-kv-shared | 1 | P11＋P10 KV norm/RoPE/store及packed交换 | 相对F-beta-shared只变P10，作为CP overlap的另一方案 | 已准备、未验收 |

P03与P10互斥；P10已含Q拼接，所以该行P04关闭。每组一套服务、一份正式客户端、一份排队profile客户端。终态、8rank库／源码／环境和命中证据核验后才精确停止本轮服务，另核8卡释放后启动下一组，不并行抢卡。

## 正确性及后续收益确认

每组先复用原有5-shot GSM8K前20题、greedy、parallel4、max512、原有accuracy>0.8门槛。不得用这个小样本替代完整准确率，也不得把随机token压测输出当精度证据。

F-control、F-beta及拟交付候选另复用相同既有评测，20-shot／100题、greedy／parallel4／max512，正式性能和profile均结束后cold flush再执行。记录原始完整回答、原解析器预测、实际prompt token数、prompt／数据集／评测源码哈希；前20题同时出现在示例中，单列该20题，并单独报告余下80题。要求实际prompt>2048以覆盖长上下文稀疏索引路径，配对解释结果，不放宽原数值或评分门槛。

组合只形成初步测量后，再选候选作同包／同代码的重复无profiler对照；不因筛选均值最低就宣称稳定收益。复测如果不用profile，将使用明确的独立重复协议、完成标记和清理守卫，不伪造8trace完成标记。最终源码15文件哈希须再次与PR实际代码内容对齐；其他GitHub CI在PR上执行。
