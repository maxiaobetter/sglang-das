# GLM prefill LightOp 兼容包

2026-11-04，在独立 `feat/glm53-prefill-fusions` 分支修改，基于 LightOp master `cf5b954074743d0ce620eb8d1ce8479615a3e259`。容器原包保留，原生数值检查已完成，P06性能断言失败；模型验收尚未完成，不能当作端到端性能已验证包。

## B2 候选

源码提交 `c3881932743ed63e2fd6c30c8474d46ac8d5c15a`，包含B1 `8c2b40a818fd`。

- P05/P09：显式 `match_sglang=True` 的RMSNorm/FP8 ABI，保持FP32 residual累加、norm输出BF16舍入和基线无scale下限的per-token量化；原默认API行为保留。
- P10：声明并允许GLM的Q head stride256；不将stride192假设套入GLM。
- P08：新增显式 `fuse_silu_mul_fp8_quant_sglang`，仅BF16 `[M,4096]`→FP8 `[M,2048]`，保留SiLU和乘法的BF16中间舍入、amax/448量化及零行规则，不套用旧融合核的最小scale。
- P06所需NaN兼容能力来自已核对的LightOp基点，本次框架需确认能力标记才允许调用。

wheel 路径：

```text
/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515/lightop_c3881932743e/dist_candidate/lightop-0.6.0+das.dtk26041-cp310-cp310-linux_x86_64.whl
```

| 项 | 值 |
| --- | --- |
| wheel大小 | 56201094 bytes |
| wheel SHA256 | `bcf4b829fb6d003da60fa8627b82f30d4e68b4c9f9fea221a082ece0eed75f41` |
| op.so SHA256 | `f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c` |
| 编译环境 | Python3.10，Torch commit a346d24b0aa21d34ebda15ba9839433496b9d887，DTK26041，CXX11 ABI=true，gfx938 |
| 编译结果 | build.rc=0，216 objects，wheel ZIP完整性通过 |
| 隔离前缀 | 本轮远端根目录下 `lightop_c3881932743e_prefix/`，只解压，没有覆盖site-packages |

全量源码使用官方setup.py构建，只在CPU build driver中固定gfx938查询结果并隐藏GPU，未改native编译路径或删除源文件。p2c8卡有其他作业，未在其上执行GPU测试。

## 验证要求

复用库已有KV gather与656-byte RoPE/store检查，GLM stride通过进程内 `Q_C_DIM=192` 调用同一fixture及断言，不改原文件或阈值；另以独立进程比较原安装包的norm→quant、SiLU→quant管线，记录命令与原始结果。KV gather已有测试已执行：5 passed、1 failed。失败发生在性能断言，81920-token shape的数值bitexact比对已通过；native237.096us，Triton145.928us，原性能要求native更快，未放宽或删除断言。P06不记为加速。656-byte fused RMSNorm/RoPE/store复用原fixture，将Q_C_DIM在进程内设为192，168组+2组GLM实际4096tokens/64heads均通过，另有原有参数校验通过。Norm/quant的16组输出与scale、shared SwiGLU的8组FP8字节及scale均与原安装包逐位一致，覆盖M=1/7/129/4096、norm宽2048/6144及有无residual；SwiGLU含小幅值和零行。原始失败保留于远端 `lightop_native_b2/`。结构化结果见 `lightop_native_results.json`，数值通过与P06性能失败分开记录。

按每token至少读取656B cache+4B索引、写1152B BF16结果估算，native有效带宽约 **626 GB/s**，Triton约 **1017 GB/s**。这是逻辑流量/API时间估算，包含随机gather和调用影响，并非PMC实测HBM带宽。

原生正确性通过后，还需使用同一B2包、全部优化关闭的模型匹配对照，再分别开启P05/P06/P08/P09/P10。B1/B2换包不是单项融合收益，原B0继续保留。

提交前已按LightOp quality-gate v2.0.3实际code-quality/governance策略运行本地lint（Ruff0.12.1匹配镜像）；Python语法与diff check通过。security和完整CI留给对应仓库，lint/编译成功不替代GPU数值正确性。
