# DPSK：GLM5.3 当前组合性能测试接手说明

2026-11-05，北京时间。交接范围：继续当前普通 IFB prefill 配置的性能筛选和少量最终确认。MegaMoE 暂缓，不集成、不构建、不插队。

## 1．用户要求与当前队列

- 全部优化代码和结果继续提交到 `feat/glm53-prefill-opt`，不要拆新分支。到旧分支的[汇总PR #52](https://github.com/maxiaobetter/sglang-das/pull/52)保持草稿，标题与正文使用中文。
- 以性能为主，精度检查仍保留；相对匹配参考多错1～2题先不作为淘汰条件。保留原始分数、逐题回答和差异，不修改解析器或原 `accuracy > 0.8` 阈值，不把小样本容差称为完整精度验收。
- 不新增仓库单测，不本地跑完整测试套件；代码提交前按项目实际 CI 配置执行 lint、Python 语法检查及 `git diff --check`。

**已有自动队列正在执行，不要重复启动。** 本地控制器 PID `2350`、启动时间 `2026-11-05 09:35:09`，命令 `python3 -u /tmp/glm_prefill_run_speed_search.py`。PID 只用于定位，接手时重新核对命令与启动时间。控制器会串行完成下表三组，逐组归档、按PID／启动时间清理并核验8卡释放，最后退出；它不会自动启动下一节的确认实验。

| 实验 | 配置，均使用已有 MQA beta | 状态快照 |
| --- | --- | --- |
| R06-candidate | KV融合／打包CP交换＋共享专家重叠＋Q投影重叠 | 已启动，GSM20为19/20，正在四轮性能测试 |
| R07-candidate | CP重叠＋Q直写＋共享专家重叠＋Q投影重叠 | 排队，未启动 |
| R08-candidate | 仅beta，其余新增开关全0 | 排队，本轮参考 |

每组原种子01／02各跑两次，共四轮；正式性能与评测分开，之后做同口径100题评测。**四轮复用同一服务，不是四次独立重启。** 状态快照会过时，以实时日志、实际进程及退出码为准；不要因为状态文件仍写“加载”就重复启动。

## 2．机器、容器、源码和包

| 项 | 固定值 |
| --- | --- |
| 连接 | Mac → `ssh sothisai2` → `ssh ai01` → `ssh p2c1`／`ssh p2c9` |
| 节点／rank | p2c1为节点rank0，p2c9为节点rank1，每节点4卡，共8卡 |
| 容器 | 两节点均为 `sglang_glm_mx` |
| 镜像 | `42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-scalex40-20261027-1047-glm53w4a8-fix` |
| 模型 | `/public/share/gxntljsb/models/GLM-5.3-Channel-FP8-w8a8` |
| 运行根目录 | `/public/home/maxiao1/scripts/glm5.3/fp8/runs/prefill_opt_20261104_171515` |
| 实际服务源码 | 根目录下 `candidate_delivered_7684f58399ad70de47de7678897f7ea5aab6107e` |
| LightOp隔离前缀 | 根目录下 `lightop_c3881932743e_prefix` |
| LightOp op.so SHA256 | `f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c` |
| 原FlashMLA SHA256 | `b0ea2f7f108da6ba22e951c3e795a1ba8a8f95bd70a6843758cb434e1296d66e` |
| 原DeepGEMM SHA256 | `79325b382a56de60f69cfaf8f75fe2b725d2092c002d1cf67bd1248f1537af7a` |
| 服务地址／端口 | `http://11.16.100.199:31023`；分布式初始化 `11.16.100.199:31500` |

源码是冻结的容器原安装副本叠加15个本分支Python文件；不是把整个容器源码等同于Git分支基点。后续提交目前只改文档。当前及后续确认完成前，不修改这15文件或隔离包，否则要另建实验身份。每组 `deployed_source.json` 和两节点 `*_runtime.json` 保留真实路径、版本及哈希；以实际加载为准。不要重新换包、GEMM选优或启用skinny。

新服务脚本位于根目录 `serve_<实验名>.sh`，当前为 `serve_R06-candidate.sh`、`serve_R07-candidate.sh`、`serve_R08-candidate.sh`。它们已经显式设置隔离 `PYTHONPATH`、环境和完整启动参数；不要套用会清空 `PYTHONPATH` 的其他旧脚本。

固定配置：普通IFB、eager，TP8／CP8／EP8、CPv2 interleave、LayerSplit关闭；输入112640、缓存107008（95%）、新输入5632、输出1，全局BS8，每轮32请求，chunk32768，EAGLE5／1／6。无EPLB映射。每轮flush缓存并按原脚本预热，逐请求核对实际输入／缓存／输出及32/32成功。

## 3．先接管观察现有队列

以下命令在**当前这台Mac**运行；`/tmp`中的辅助程序通过三跳SSH操作集群，不要在Linux上直接运行含Mac绝对路径的本地控制器。

```bash
ps -p 2350 -o pid,lstart,args
cat '/Users/better/Desktop/SGLang相关/sglang_dcu_github_origin/dcu-sglang/profiles/glm53-prefill-opt-20261104/speed_search_status.json'
python3 /tmp/glm_prefill_quick_progress.py R06-candidate
python3 /tmp/glm_prefill_long_progress.py R06-candidate
```

按状态替换为R07／R08。`quick_progress`查看正式性能和服务末尾；`long_progress`查看100题评测。模型主权重和NextN两次加载，启动常见约9～10分钟，不因一次观察超时而重启。若控制器报告错误，检查同一客户端PID／启动时间、退出码和原始日志，先保留现场；不要重新运行整条队列。

本地结果目录：

```text
/Users/better/Desktop/SGLang相关/sglang_dcu_github_origin/dcu-sglang/profiles/glm53-prefill-opt-20261104/
  speed_search_plan.json
  speed_search_status.json
  R06-candidate/、R07-candidate/、R08-candidate/
```

根目录中每组的关键文件：`client.log`、`client.rc`、`BENCH_COMPLETE`、`verified.json`、`repeat1..4/bs8.jsonl`、`additional_accuracy.rc`、`gsm8k_20shot_100/{result.json,all_outputs.json,actual_requests.jsonl}`、两节点runtime／verification／hca，以及清理／释放记录。

控制器中 `record_search` 会严格核验四轮请求、8rank身份、100题实际请求和清理证明，生成优化分支下的 `SPEED_SEARCH_results.json`／`SPEED_SEARCH.md`。**自动写报告不等于已提交Git**，接手后要查看diff、lint再提交。报告中的“本轮三组最优”只是初筛，不能写成全部历史候选的最终最优。

辅助程序已保存快照，避免只依赖临时目录：

```text
/Users/better/Desktop/SGLang相关/sglang_dcu_github_origin/dcu-sglang/profiles/glm53-prefill-opt-20261104/dpsk_handoff_20261105/
  helpers/                       # 22个辅助文件
  helpers_SHA256.json
  helpers_snapshot.tar.gz
  speed_search_plan.json
```

归档SHA256：`b48f1bd0a6c2600b1f735b14ba296b6abad1b80e7de3ec32631e553942cd7ca0`。优先使用当前已有文件；若恢复快照，先比哈希并检查期间的新改动，不覆盖新文件。快照中的Mac绝对路径保持原样。

## 4．三组完成后的最小确认

不要重跑全矩阵。R08的同源码／同包beta参考可直接作为紧邻的A前，随后串行做：

| 次序 | 配置 | 目的 |
| --- | --- | --- |
| A前 | 已完成的R08 beta-only | 紧邻前置参考，避免重复跑一遍 |
| B | 当前领先组合 | 独立重启后确认收益 |
| C | B只切换Q投影重叠，其余完全相同 | 判断Q重叠是否拖慢；若B已关闭，则C开启 |
| A后 | beta-only，与R08同源码／同包 | 判断基线漂移 |

**B的候选要包含R03的原配置**（beta＋KV融合＋shared重叠、无Q投影重叠），不能只在R06／R07中选。R03两轮均值4693.77ms、93/100；其与旧beta有约8小时间隔，旧差值不作Q投影净收益证明。R01的CP／Q／shared复测4807.33ms、96分也保留。初筛4673.05ms（降低6.88%）未在R01重现，不能拿初筛最低值当最终稳定收益。

B若来自R06，C对应R03的开关；B若来自R07，C对应R01的开关。若R03更快，则B为R03，C为加上Q投影重叠的R06配置。两条路线结果接近时，可把另一候选纳入同一确认窗口；不要无限扩矩阵。

每个新会话固定原种子 `2026110401`／`2026110402`、两轮无profiler，沿用相同flush／预热和请求；保留逐轮TTFT、新输入吞吐与分数。若A前／A后漂移接近候选收益，补一轮相邻配对再判断，不强判优胜。R08与后续确认若隔太久或环境改变，重新建立A前，不能冒充相邻对照。

新建未使用的实验ID及空目录，建议B=`R09-candidate`、C=`R10-candidate`、A后=`R09-control-b`；先确认不存在，不复制旧实验整个目录或PID文件。只复制选定父配置的服务脚本和部署manifest，更新case／开关／脚本SHA及两轮协议。C只改 `SGLANG_GLM_PREFILL_Q_PROJ_OVERLAP` 一行，并同步manifest的开启项；其余开关、包和参数不变。

可复用以下**两轮**辅助程序，逐组执行并等待终态；不要拿四轮专用的`run_speed_case`／`schedule_search_accuracy`／`stop_search`直接跑新的两轮实验：

```bash
# Mac；先按上文准备新实验空目录、脚本与manifest
case_id=R09-candidate
python3 /tmp/glm_prefill_preflight_repeat.py "$case_id"
python3 /tmp/glm_prefill_launch.py "$case_id"
python3 /tmp/glm_prefill_run_case.py "$case_id"
# 等client.rc=0、BENCH_COMPLETE=0、verified.json含两轮后
python3 /tmp/glm_prefill_capture_runtime.py "$case_id"
python3 /tmp/glm_prefill_verify_delivered.py "$case_id"
python3 /tmp/glm_prefill_schedule_repeat_accuracy.py "$case_id"
# 等additional_accuracy.rc=0，先完整归档再清理
# 不调用fetch_repeat冒用父配置旧trace作为新组合证明
python3 /tmp/glm_prefill_stop_repeat.py "$case_id"
python3 /tmp/glm_prefill_free_resources.py "$case_id"
```

归档需覆盖整个实验的命令、原始数据／回答、部署与runtime证据和服务日志；清理后的PID快照／退出结果／显存释放观察另补入本地归档。现有`fetch_search`只支持R06～R08四轮，后续两轮需按新ID／轮数适配归档和结果读取，保留严格核验。不要伪造`profile_reference`或`PROFILE_COMPLETE`来满足旧读取器。

## 5．最后profile与交付

只给最终少量候选独立抓新profile：与正式性能分开，使用固定第三种子 `2026110403`，同BS8和32K chunk，全部8rank trace校验并保留manifest。旧父配置trace不能替代新组合实测。可参考现有`glm_prefill_profile_case.py`；两轮／无profiler的清理器不能直接清理已抓profile的会话，需按真实profile终态采用相应清理与归档流程。

重点确认主模型 `step[EXTEND bs=6 toks=32768]`，不要把客户端BS8与单步bs6混淆；核对Q投影／shared／CP或KV融合实际命中、主步时间和关键依赖。组合有多条计算流，不可沿用“唯一非主流GEMM就是Q流”的单项分析假设；按显式scope／关联ID识别。不把跨流累计时间或交集直接当作实际节省。

结果至少列出：配置和全部开关、两轮／四轮各自数据及均值、相同种子的比较、相对beta及历史base的变化、100题／非示例80题分数和逐题变化、源码／包哈希、命令与日志／trace路径、资源释放情况。分别写“最快测量配置”和按1～2题容差后的推荐；仅覆盖本次IFB BS8／95%命中／32K场景，PD和其他BS单列。

本地优化工作区：`/Users/better/Desktop/SGLang相关/sglang_dcu_github_origin/glm53-prefill-opt`。源码15文件仍冻结在7684快照；记录更新提交到同一个优化分支。CI以`.github/workflows/lint-hcu.yml`和`.pre-commit-config.yaml`为准，相关GitHub CI目前排队，不能记为通过；PR继续中文、草稿。

资源规则：每组启动前重新核8卡／KFD进程／端口，停止只匹配本轮PID及启动时间；禁止宽泛`pkill`、重启Docker／节点，或抢占其他作业。p2c8曾有无关作业，不使用其卡。模型加载、单次轮询超时、文件已存在都不代表测试完成；以真实终态和核验证据为准。
