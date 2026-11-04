# LightOp B2 FP8 MQA 既有数值检查

复用 `test/mqa_logits/test_mqa_logits.py` 原有FP8 numeric-only模式、mask逐位相等断言和 `calc_diff < 1e-3`，没有改原文件或放宽阈值，没有新增仓库单测。

逐进程设置beta0/1、普通/CP mask，M=16/128、N=4096/131072、32heads、D128，共16组，4个子进程退出码及总退出码全部0。最大打印calc_diff为 **7.45e-4**。既有测试的reference仍为BF16原输入，包含FP8量化误差；未把它改成更容易通过的reference。具体命令、每组日志、原文件SHA256和设置均留存。

这只是两个MQA后端的既有原生数值检查，不能替代TopK完全相等或模型准确率，也不是性能结果。B2-beta独立模型组 `P01-B2` 仅已准备，尚未启动，不能把历史P01原包成绩归给新包。

隔离B2的native文件SHA256为 `f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c`。短数值调用没有另外采集子进程maps，保留了精确PYTHONPATH，并核验前缀实际.so及原测试文件；B2模型运行的8rank实际加载哈希已另行核验。

远端本轮根目录的 `mqa_existing_b2/`；本地原工作区 `profiles/glm53-prefill-opt-20261104/lightop-build-b2/mqa_existing/`。测试退出后另行确认8卡2MiB才启动P09。结构化结果见 `MQA_native_results.json`。
