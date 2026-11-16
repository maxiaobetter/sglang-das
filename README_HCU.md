# <div align="center"><strong>SGLang HCU 0.5.18</strong></div>

> 本文档适用于 `release/20260825_v0.5.18` 分支及其 HCU 适配版本。
> 模型部署示例和硬件适配说明请参考 [HYGON-AI/inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das)。

> 已验证镜像：`42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-20260831-0154`。该镜像使用 Torch 2.11.0、`sglang 0.5.18` 和 `sglang-kernel 0.4.6.post1`；BW1000/天龙网卡环境应优先使用带 `sf` 标识的镜像。

## sglang_hcu简介
SGLang是一个用于大型语言模型和多模态模型的高性能服务框架，旨在在从单个GPU到大型分布式集群的各种设置中提供低延迟和高吞吐量的推理，我们基于开源社区做了HCU平台的适配和针对性的优化。
其核心功能包括：快速运行时：通过RadixAttention提供高效的服务，用于前缀缓存、零开销CPU调度器、预填充解码分解、推测解码、连续批处理、分页注意力、张量/流水线/专家/数据并行性、结构化输出、分块预填充、量化（FP4/FP8/INT4/AWQ/GPTQ）和多LoRA批处理。
广泛的模型支持：支持各种语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等）、嵌入模型（e5-Mistral、gte、mcdse）、奖励模型（Skywork）和扩散模型（WAN、Qwen-Image），易于扩展以添加新模型。与大多数Hugging Face模型和OpenAI API兼容。
强化学习和训练后主干：SGLang是一个经过验证的全球推广后端，具有原生强化学习集成，并被AReaL、Miles、slime、Tunix、verl等知名训练后框架采用。

## GLM-5.3 HiSparse（待设备验证）

本实现统一基于 `maxiaobetter/sglang-das` 的 `glm5.3-w4a8` 分支，W4A8 和 FP8 模型共用 HiSparse 及单层 NextN/MTP 的缓存管理代码。权重量化由模型配置及现有算子分发决定，HiSparse 不按权重量化类型拆分实现。后端按 **KV cache dtype** 选择：`fp8_e4m3` 使用 `flashmla_kv`，`bfloat16` 使用 `flashmla_sparse`。FP8 KV 沿用 LightOp 的打包格式，每 token 为 512B latent、16B scale 和 128B BF16 RoPE，共 656B；host/device 之间按字节复制，不重新量化。

在已有的 PD **decode** 启动命令中增加：

```bash
--enable-hisparse \
--disable-radix-cache \
--kv-cache-dtype fp8_e4m3 \
--page-size 64 \
--dsa-prefill-backend flashmla_kv \
--dsa-decode-backend flashmla_kv \
--hisparse-config '{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2,"swap_in_block_size":256}'
```

省略两个 DSA backend 参数时，HCU 会按 KV dtype 选择上述后端。使用 BF16 KV 时将 dtype 改为 `bfloat16`，两个 backend 改为 `flashmla_sparse`。`top_k` 必须与模型的 DSA top-k 一致；`device_buffer_size` 和 `host_to_device_ratio` 应按上下文、并发以及 host/device 内存容量设置。HCU 的换入及预取复制 kernel 将 block size 上限限制为 256。

P 实例继续使用原有 CP/EP 配置，无需启用 HiSparse。P/D 两侧必须使用相同的 KV dtype、page size 和打包格式；FP8 P 侧可继续使用现有的 `flashmla_auto` prefill 路径。D 实例可沿用 DP/EP 配置，`dcp_size` 必须为 1。

普通 decode 和 GLM 单层 NextN/MTP 均已接入代码。MTP 在上述 D 实例参数之外保留原有的并发限制，并增加以下参数（例如每轮 verify 6 个 token）：

```bash
--speculative-algorithm NEXTN \
--speculative-eagle-topk 1 \
--speculative-num-steps 5 \
--speculative-num-draft-tokens 6 \
--disaggregation-transfer-backend mooncake
```

MTP 要求明确设置全局 `--max-running-requests`，以便预算每个 DP worker 的缓存管理元数据及 PD 预分配请求槽位；例如 DP32、每 DP 最多 4 请求时使用 128。当前仅覆盖单 draft 层、线性链（topk=1）、固定 verify 宽度 2–16、PP=1、DCP=1；不覆盖树状、多层或自适应步数。`width * index_topk` 上限为 32768，hot buffer 须按 64 对齐且不超过 32768。P 侧同样启用原有 MTP，保留 CP/EP 配置。

当前一个 D worker 同时执行一轮 verify；两批重叠（TBO）和 PDMux 需要独立事务工作区，本版会明确拒绝这两个组合。

目标模型每层将整轮所有 verify 行的 Top-K 联合换入，保留各行原始顺序及因果有效范围，并在 verify 后只备份接受的 token；拒绝的 KV 不写入 host。每请求另预留 `ceil(width * index_topk / 64) * 64` 个物理 token 的工作区，与 hot buffer、write page 一并计入 admission 容量。以 width=6、top-k=2048、hot=4096 为例，每请求共需 16448 个目标物理 token 槽位。

Draft 层保留完整逻辑地址空间的 dense KV，容量按 `host_to_device_ratio` 扩展，page size 保持 64；其 KV 显存及 verify 元数据已计入池预算。PD 传输分别携带目标 host 页 ID 和 draft device 逻辑页 ID，目标 KV 进入 DRAM，draft KV 进入 VRAM。不同权重量化版本沿用同一路径，不改变 W4A8/FP8 的数学计算。

实现参考此前 HiSparse 分支的 `7ead84f7f646`（PD/graph/HIP）、`bc2b06fafa40`（同轮缓存复用）及 `bda41d6895e7`（多行换入、accepted-only 备份）。MTP 共享索引预取参照官方 `2f7dd856a0346fdb9a474e4362fcc0f3cdb67e1a`：共享组首层执行 union lookup、更新 hot/scratch 布局并记录换入计划，后续共享层在预取流上按相同槽位偏移搬运各自层的 KV，等待对应事件后复用首层页表，不重复规划。各层 KV 内容及存储仍独立；本轮 verify KV 在各层固定 write page 生成，不从 host 预取。计划容量为 `width * index_topk`，已计入显存预算。`SGLANG_DISABLE_HISPARSE_PREFETCH=1` 可在启动时回退逐层同步换入，不支持运行中切换。由于当前 AOT transfer 不屏蔽负索引，接受 token 的备份使用带长度掩码的专用 JIT kernel，不传入 `-1` 哨兵。

此补丁尚未完成 HCU 容器中的算子对拍、PD 传输和准确率验收；CUDA graph 也需要验证更换输入后的重放。应先核对实际安装的 LightOp、FlashMLA、sgl-kernel 路径、版本、哈希及 ABI，再抓取各并行组实际工作 rank 的完整 profile。静态检查通过不代表上述运行验证通过。

## 使用源码编译方式安装
提供2种环境准备方式:

1. 使用 0.5.18 HCU 基础镜像，正式镜像当前基线为 Python 3.10、PyTorch 2.11.0 和 DTK 26.04（`DTK-26.04-rc4`）。如果要复现 PR HCU wheel CI，则按 workflow 使用 PyTorch 2.10.0。

2. 如果使用现有 Python 环境，请先安装与 Python、DTK 和系统架构匹配的 PyTorch 2.11.0 HCU wheel；复现 PR wheel CI 时使用 2.10.0。HCU PyTorch wheel 可从 [OpenDAS PyTorch 仓库](https://cancon.hpccube.com:65024/4/main/pytorch) 获取。安装示例：
```shell
pip install <匹配的torch-2.11.0 HCU wheel>
pip install setuptools wheel
```

### 源码编译安装
```shell
git clone https://github.com/HYGON-AI/sglang-das.git
cd sglang-das
git checkout release/20260825_v0.5.18
```
安装依赖:
```shell
pip install -r requirements_hcu.txt
```

### 源码编译安装
进入仓库根目录后：
```shell
# 编译并安装 HCU sgl-kernel
cd python/sglang/kernels/aot
python setup_hip.py install
cd ../../../..

# 方式一：构建并安装 sglang wheel
cd python
python -m build --wheel --no-isolation -Cfeatures=all_hip
pip install dist/sglang-*.whl --no-deps
cd ..

# 方式二：以 editable 模式安装源码
pip install -e "python[srt]" --no-deps --no-build-isolation
```

### 运行基础环境准备
1、使用上面所选的 Python 3.10、PyTorch 版本和 DTK 26.04 HCU 基础环境。

2、根据上面所选的 PyTorch 版本（正式 0.5.18 镜像使用 2.11.0，PR wheel CI 使用 2.10.0）、Python 3.10、DTK 26.04 和系统架构下载对应的 HCU 依赖包:
- flash_attn: [https://cancon.hpccube.com:65024/4/main/flash_attn](https://cancon.hpccube.com:65024/4/main/flash_attn)
- flash_mla: [https://download.sourcefind.cn:65024/4/main/flash_mla](https://download.sourcefind.cn:65024/4/main/flash_mla)
- lightop: [https://download.sourcefind.cn:65024/4/main/lightop](https://download.sourcefind.cn:65024/4/main/lightop)
- triton: [https://cancon.hpccube.com:65024/4/main/triton](https://cancon.hpccube.com:65024/4/main/triton)
- vllm: [https://download.sourcefind.cn:65024/4/main/vllm](https://download.sourcefind.cn:65024/4/main/vllm)

### 注意事项
+ 若使用 pip install 下载安装过慢,可添加源:-i  https://mirrors.huaweicloud.com/artifactory/pypi-public/simple

## 验证
- python -c "import sglang; print(sglang.\_\_version__)",

## 手动无人值守大模型测试

仓库提供独立 workflow `Manual HCU Unattended Model Test`（文件：`.github/workflows/hcu-manual-model-test.yml`），用于在 HCU runner 上手动启动长时间模型验证任务。它只支持 `workflow_dispatch`，不会在 `pull_request`、`push` 或 `schedule` 时自动触发，因此不属于 PR gate。

使用方式：进入 GitHub Actions，选择 `Manual HCU Unattended Model Test`，点击 `Run workflow`。GitHub 页面上的分支下拉框决定默认测试 ref；也可以通过 `test_branch` 输入分支、tag、ref 或 SHA 覆盖。默认 suite 为 `nightly-hcu-accuracy`，可通过 `suite` 改成 `nightly-hcu`、`nightly-hcu-vlm`、`nightly-hcu-4-gpu` 等已注册 HCU suite。若只跑单个文件，填写 `include_file`，例如 `test/registered/hcu/accuracy/bw1100/test_gsm8k_eval_hcu.py`。

常用输入包括：`timeout_per_file`（默认 4200 秒，沿用现有 HCU accuracy 长测配置）、`auto_partition_id` 和 `auto_partition_size`（必须成对填写）、`continue_on_error`（默认 true，表示 `run_suite.py` 内部尽量继续跑后续文件，但 workflow 仍会在最终失败时显示失败）、`runner_label`、`container_name` 和 `image`。`model_name` 会传给 `SGLANG_HCU_GSM8K_MODEL`、`SGLANG_HCU_MMLU_MODEL` 和 `SGLANG_TEST_DEFAULT_MODEL_NAME`，可用于测试其它本地模型路径，例如 `/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/Qwen2.5-7B`。`run_suite.py` 尚无 `--model-name` 过滤参数，因此这里通过测试文件已支持的环境变量选择模型。

快速验证其它模型时，可填写 `model_name`，并将 `eval_num_examples` 设为较小值（例如 `10`）；也可用 `include_file` 只跑 `test/registered/hcu/accuracy/bw1100/test_gsm8k_eval_hcu.py` 或 `test/registered/hcu/accuracy/bw1100/test_mmlu_eval_hcu.py`。如模型精度阈值不同，可通过 `gsm8k_threshold`、`mmlu_threshold` 临时覆盖。`mmlu_num_threads` 默认 128，这是在 HCU runner 上手动验证过的稳定配置。

该 workflow 默认会在测试前从所选 ref 重新编译并安装 `sgl-kernel`（`rebuild_sgl_kernel: true`），避免 checkout 出来的 `sglang` 源码与镜像里预装的 `sgl-kernel` 扩展包接口不匹配。如果明确使用镜像内完全匹配的包，可手动关闭该选项。

运行结束后，可在 workflow 日志中查看测试 ref、runner label、容器名、suite、include_file、model_name、timeout、分片配置和最终执行命令。job summary 会展示本次配置和结果；artifact `hcu-manual-model-test-<run_id>` 会上传 `summary.md`、最终命令和 `run.log`。

## PD 分离

### Requirements

```bash
pip list | grep mooncake-transfer-engine
```

### Usage

#### Normal

**加载环境变量：**

```bash
export MC_TOPO_FILE_FORCE=./mc_topo.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
#export MC_IB_GID_INDEX=0 #Roce网络需要设置
```
mc_topo.config

```YAML
0000:9f:00.0 mlx5_2 hip:0
0000:57:00.0 mlx5_3 hip:1
0000:5e:00.0 mlx5_4 hip:2
0000:05:00.0 mlx5_5 hip:3
0000:e5:00.0 mlx5_6 hip:4
0000:c1:00.0 mlx5_7 hip:5
0000:cc:00.0 mlx5_8 hip:6
0000:b1:00.0 mlx5_9 hip:7
```
**DeepSeek-R1-Channel-INT8 模型示例**

##### prefill
```bash
python -m sglang.launch_server \
  --model-path DeepSeek-R1-Channel-INT8 \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --disaggregation-mode prefill \
  --host ${prefill_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 2 \
  --pp-size 4 \
  --mem-fraction-static 0.9 \
  --attention-backend hcu_mla
```

##### decode
```bash
python -m sglang.launch_server \
  --model-path DeepSeek-R1-Channel-INT8  \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --disaggregation-mode decode \
  --host ${decode_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 8 \
  --mem-fraction-static 0.9 \
  --attention-backend hcu_mla \
  --dtype bfloat16 \
  --quantization slimquant_marlin
```
##### 启动路由：
```bash
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://${prefill_ip}:30000 --decode http://${decode_ip}:30000 --policy round_robin --port 30002
```
##### 验证输出结果
在另一个终端中，使用以下命令验证输出结果：
```bash
curl -X POST http://localhost:30002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "介绍一下深度学习的发展",
    "max_tokens": 300,
    "temperature": 0
  }'
```
#### low_latency （使用deepep）
prefill部分同上面normal部分的prefill

##### decode
```bash
# deep_ep
#export ROCSHMEM_DISABLE_HDP_FLUSH=1 #xdp使用
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_ENABLE_LL_DISPATCH_OPT=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config   // 与上面文件一致

# mooncake
export MC_TOPO_FILE_FORCE=./mc_topo.config      // 与上面文件一致
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
#export MC_IB_GID_INDEX=0 #Roce网络需要设置
```
topo.config

```YAML
0000:9f:00.0 mlx5_2 2
0000:57:00.0 mlx5_3 3
0000:5e:00.0 mlx5_4 4
0000:05:00.0 mlx5_5 5
0000:e5:00.0 mlx5_6 6
0000:c1:00.0 mlx5_7 7
0000:cc:00.0 mlx5_8 8
0000:b1:00.0 mlx5_9 9
```
单机ep8dp2部署示例
```bash
python3 -m sglang.launch_server --model-path DeepSeek-R1-Channel-INT8 \
--disaggregation-mode decode --quantization slimquant_marlin \
--kv-cache-dtype fp8_e4m3 --host ${decode_ip} --port 30000 --trust-remote-code \
--dist-init-addr ${decode_master_ip}:5000 --nnodes 1 --node-rank 0 --dtype bfloat16  \
--tp-size 8 --dp-size 2 --mem-fraction-static 0.85 \
--moe-dense-tp-size 1 --enable-dp-lm-head \
--attention-backend hcu_mla --enable-dp-attention --moe-a2a-backend deepep  \
--ep-size 8 --deepep-mode low_latency \
--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```
多机ep16dp16部署示例
```bash
#node1 作为主节点
python3 -m sglang.launch_server --model-path DeepSeek-R1-Channel-INT8 \
--disaggregation-mode decode --quantization slimquant_marlin \
--kv-cache-dtype fp8_e4m3 --host ${node1_ip} --port 30000 --trust-remote-code \
--dist-init-addr ${node1_ip}:5000 --nnodes 2 --node-rank 0 --dtype bfloat16  \
--tp-size 16 --dp-size 16 --mem-fraction-static 0.85 \
--moe-dense-tp-size 1 --enable-dp-lm-head \
--attention-backend hcu_mla --enable-dp-attention --moe-a2a-backend deepep  \
--ep-size 16 --deepep-mode low_latency \
--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

#node2
python3 -m sglang.launch_server --model-path DeepSeek-R1-Channel-INT8 \
--disaggregation-mode decode --quantization slimquant_marlin \
--kv-cache-dtype fp8_e4m3 --host ${node2_ip} --port 30000 --trust-remote-code \
--dist-init-addr ${node1_ip}:5000 --nnodes 2 --node-rank 1 --dtype bfloat16  \
--tp-size 16 --dp-size 16 --mem-fraction-static 0.85 \
--moe-dense-tp-size 1 --enable-dp-lm-head \
--attention-backend hcu_mla --enable-dp-attention --moe-a2a-backend deepep  \
--ep-size 16 --deepep-mode low_latency \
--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

##### 启动路由：
```bash
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://${prefill_ip}:30000 --decode http://${decode_ip}:30000 --policy round_robin --port 30002
```
##### 验证输出结果
在另一个终端中，使用以下命令验证输出结果：
```bash
curl -X POST http://localhost:30002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "介绍一下深度学习的发展",
    "max_tokens": 300,
    "temperature": 0
  }'
```

## Known Issue
- 无

## 参考资料
- [HCU 模型部署 Cookbook](https://github.com/HYGON-AI/inference-cookbook-das)
- [README_ORIGIN](README_ORIGIN.md)
- [SGLang 上游项目](https://github.com/sgl-project/sglang)

## License

本仓库基于 [SGLang](https://github.com/sgl-project/sglang) `release/20260825_v0.5.18` 版本进行 HCU 平台适配和优化，上游项目采用 Apache License, Version 2.0。

Hygon Information Technology Co., Ltd. 对 HCU 适配、修改和新增贡献部分同样采用 Apache License, Version 2.0。

本仓库保留上游 SGLang 项目的版权声明和许可证条款，详见 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`。
