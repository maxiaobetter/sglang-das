# 新增入口与回退路径复核

复核对象是最终快照7684f58399的15个改动Python文件；后续提交目前仅改记录。静态复核补充运行证据，不代替其他模型／平台测试。

- 新增环境开关默认全部关闭。attention的新增入口由HCU＋`GlmMoeDsaForCausalLM`标记、普通non-speculative eager extend、DSA FlashMLA prefill等门控限定；compile／capture／deterministic／TBO／DCP／LoRA等不匹配路径保留回退。P03的复制CP通信组按显式环境开关创建，其他模型未作为开启该开关的验收范围。
- P12只在确实运行indexer的层启动独立Q-B投影，且linear不得带output gather或LoRA。`should_run_indexer`是纯布尔判断，保留原有shared层不读缺失权重的策略；未放宽成“prev_topk为None就运行所有层”。P12默认关闭，单项未确立收益。
- P03复用每device一条通信stream，保留launch／join及tensor生命周期，CPv2按logical／physical长度填充、截断并重排。P04只在BF16权重／输入及受支持形状下直写Q，保留scale和修正路径；不改FP8 GEMM选优。
- P05保留BF16 normalized Q供indexer，量化值／scale明确传给Q-B。compressed-tensors的tuple入口原已存在；新增能力探测仅接受动态channel FP8、FP32 weight scale、非AITER／LoRA的匹配linear。原有uint8权重allocator tuple走其原block路径，不把第三项输出buffer当成新增dtype。`apply_fp8_linear`扩展per-token scale接收条件由新增融合flag和明确dtype／scale shape限定；全关闭时保留原per-tensor断言。
- P09的`PrefillFP8Input`显式保存BF16与FP8两种视图，AttentionInputs仅在需要BF16的消费者解包，QKV-A消费者读取预量化值。该路径虽命中，当前包小样本得分及更广原生输入有差异，保持关闭、不进入组合；见P09，不能据静态类型正确宣称精度通过。
- P10仅替换有prefix、普通DSATokenToKVPool、FP8 656-byte布局、CPv2 interleave的writer。packed行包含scale和BF16 RoPE，收齐后按原全局physical loc写入；wrapped／compact／HiSparse／cold BF16路径保持原writer。typed stored-KV标记显式关闭后续重复写入。本组8rank使用一致的全局metadata；P03与P10显式互斥，P10已含Q拼接。
- P11复用原shared/routed事件组织，普通GLM prefill使用全模型缓存的一条shared stream，输入、输出记录stream生命周期并在主stream等待。未将其与P08量化融合记为一个开关；decode保留原门控。P08要求显式SwiGLU匹配ABI，不替换为Spark GELU或旧scale下限版本。
- P06要求明确的NaN兼容ABI，未因存在同名符号就接受旧包。实测数值与命中通过但性能回退，保持关闭。P07新增的合并Mega实验adapter已从交付移除；原Mega后端保留，P13独立包另作适用性审计，未改其他模型量化／路由语义。

上述边界的实际GLM命中、8rank源码／库／环境、数值或模型sanity与正式性能分别见各项记录。最终组合仍须完成同源码／同包对照、长上下文配对正确性及重复性能验证；不能以此静态复核替代这些门槛。其他GitHub CI在PR上执行，不新增仓库单测或默认完整测试套件。
