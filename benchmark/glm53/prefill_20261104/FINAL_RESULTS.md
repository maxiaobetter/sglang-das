# 最终源码对照与组合结果

同一最终源码15文件、LightOp B2、原FlashMLA/DeepGEMM；协议见FINAL_MATRIX.md。以下仅已完成的单配置测量，不把候选最小均值等同于稳定收益。

| ID | TTFT均值 ms | 相对F-control | 8卡新输入 tok/s | GSM20 | 长评测100／非示例80 | 主32K step ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| F-control | 5018.46 | +0.00% | 8713.11 | 19/20 | 97/100；77/80 | 3276.91 |
| F-beta | 待验 | — | — | — | — | — |
| F-beta-shared | 待验 | — | — | — | — | — |
| F-beta-comm-copy-shared | 待验 | — | — | — | — | — |
| F-beta-kv-shared | 待验 | — | — | — | — | — |

长评测复用原解析器、20-shot／100题、greedy／parallel4／max512；前20题包含在示例中，单列其正确数并重点报告余下80题。实际prompt token数和prompt／数据集／评测代码哈希保留于对应JSON及原始目录，短sanity不作完整准确率保证。

原有解析器提取最后一个整数片段；例如关闭组一题回答含15.04，其末尾04触发原整数解析的SyntaxError而记invalid，仍按原口径记错，不改分数。各配置invalid及完整回答保留，不能把invalid一概理解为空返回。各配置完整明细见 `F-*_results.json`；新flags、MQA native实际kernel、正式逐轮指标、模型metadata和8rank证据路径分别记录。未完成的适用组合、配对正确性和重复性能必须继续，不以本表代替最终验收。
