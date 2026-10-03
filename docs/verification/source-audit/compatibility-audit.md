# 官方 TaoMate-H3 单进程 MPS 适配审计

范围：只读官方源码及现有 safetensors 头，不改项目/权重、不下载模型、不运行 GPU。以下风险已直接同步 parent 与 Astra。代码改动进行中，因此所有官方行号固定指上游 commit `6b2f9984acf681e2f387188f5d1f3b26e504315b`，完整不可变审计副本为 `/private/tmp/taomate-full-audit/src/taomate_h3/`；工作库 `/Users/supreme-paca/Projects/taomate-h3-metal` 原始内容与其一致。

HF 组件源固定 `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`，位于 `/Users/supreme-paca/Projects/h3-metal/MiniMax-H3/FL2VA/{video_vae,audio_vae}/`。parent 已补齐 24 个源码并记录 `/Users/supreme-paca/Projects/taomate-h3-metal/docs/verification/component-source-receipt.json`。未把早先 h3.c 重实现当作本次适配主体。

## 必须修改的后端耦合

| 固定源位置 | 风险和适配约束 |
| --- | --- |
| `runtime/direct.py:18–19,48–73,169–201` | TP2×Ulysses2/4、WORLD_SIZE 校验、CUDA set_device、NCCL、新 group、barrier/broadcast 全写死。单进程必须显式 TP1/Ulysses1/device=mps，不能用假 NCCL group 或仅替换 device 字符串。 |
| `distributed/parallel.py:35` | 目前 group=None 被拒绝。只对 world_size=1 允许无 group，多卡检查要保留。 |
| `distributed/collectives.py:12–49`、`distributed/ulysses.py:72–81` | 即使 world1 仍要求 dist 初始化并调用 collective；需 world1 identity/no-op。`merged_qkv_sequence_to_heads:138` 已有 world1 快路，但 ordinary packed QKV 和 heads_to_sequence 仍会进入 _swap。确认 in-place 与返回副本的所有权语义。 |
| `distributed/linear.py:66–69,177–181` | Column gather 与 Row reduce 即使 world1 仍走 collective。完整 QKV/FC1 merged weights 顺序必须保持；world1 不是保留 TP2 的半宽权重。 |
| `model/pipeline.py:57,70–83`、`model/dit.py:123–160` | pipeline 默认 W8A8 policy 会调用 vLLM/CUTLASS；MPS 构造必须显式传 None，不能等 apply_acceleration 再禁用。BF16 模块原生已有完整路径。 |
| `inference/acceleration.py:28–49` | 默认启用 Triton、FA3、CUTLASS workspace。MPS 应选择官方 eager BF16 算子、torch SDPA，保留 LoRA BF16 materialization/可移植 static caches。receipt 应诚实区分 bf16 MPS 与官方 selective W8A8 CUDA。 |
| `streaming/attention_hook.py:42–50,85–92` | 强制 FA3；MPS adapter 用 PyTorch SDPA，保持 NHD↔BHND 转置、scale、dropout=0、noncausal 与完整可见 KV 单次 softmax。 |
| `streaming/cache.py:20–23,45–48,70–74` | contract 虽从模型计算 heads，却仍保留 device_type=cuda 默认。单卡应 56×128 heads、BF16、实际 model device。 |
| `streaming/runtime.py:469–474,569,644–653,696–702` | CUDA device/events/synchronize 和 CUDA float64 telemetry。MPS 用同步的后端中立计时，并保持 noisy/clean/total call 计数。CPU 双精度时间值不必转设备；不能把 float64 tensor 创建到 MPS。 |
| `model/pipeline.py:104,174,182,263,284,310,347,349` | text offload empty_cache、decode error gather、WORLD group、非 DP 保存分支 broadcast 都有后端耦合；不只 DP 代码要 guard。 |
| `model/pipeline.py:185–192`、`model/output.py:553–570` | 无条件开启 Triton VAE exact fusions 和 temporal-window DP；后者 world1 会直接抛异常。单进程禁用这些 CUDA/多卡优化，但保留原生 spatial/temporal tiling 和 one-shot latent timeline 解码。 |
| `cli.py:125–140,159–190,219`、`teacher.py:401–426` | CLI 强制 acceleration deps、torchrun、CUDA env，临时目录固定 `/dev/shm`；teacher 自己也初始化 CUDA/NCCL。MPS 要覆盖两个入口且使用系统 tempfile，不能只让 stage3 worker 支持 MPS。 |
| `runner.py:340–342` | FPS 消费 `cuda_seconds`，改计时后必须同步 key/语义；不可把异步 enqueue 时间当执行时间。 |

## Float64 与 import-time 依赖

- `model/packed_sequence.py:31–39,75–86,132–134,185–191` 的 float64 grid 是 CPU 几何构造；`model/denoise.py:154–157` 已在 transfer 时指定 float32，这是应保留的正确边界。不要粗暴把所有 CPU grid 提前改成 BF16，也不要先 `.to('mps')` 再 cast。`streaming/runtime.py:473` 才是明确设备端 float64 的硬编码风险。[PyTorch MPS 文档](https://docs.pytorch.org/docs/main/notes/mps.html) 说明 double tensor 无法在 MPS 创建/复制，CPU fallback 环境开关也不能绕过此限制。
- `inference/fused_kernels.py:52–585` 的 Triton 顶层导入由 ImportError 保护，缺 Triton 本身可安全导入；启用函数会拒绝。不要在 MPS 调用 enable fusions：`layers.py:113–114` 等部分融合选择只看全局 flag，没有 is_cuda 二次防护。
- `layers.py:39–48` 的 FA3 import 是启用时才触发；`streaming/attention_hook.py:46` 则构造 hook 就触发。两处都需后端分支。
- `text_encoder.py:32–47` 懒加载 Transformers Qwen3VLConfig/TextRMSNorm/TextRotaryEmbedding/apply_rotary_pos_emb，`pyproject.toml:10–24` 的 Transformers=4.57.6、tokenizers=0.22.1、diffusers=0.35.2 是当前来源契约，不宜随意替换 embedding 实现。
- bundled VAE import 使用 diffusers、torchvision/PIL、numpy、PyYAML、safetensors，以及 torch.distributed 模块；单进程正确 state 下不要求初始化 dist。parent 已验证两组件 import 和 state；这并不等于真实 decoder 算子已在 MPS 跑通。

## 必须守住的 source behavior

**Noise / schedule**

- `model/pipeline.py:108–139,385–405` 使用 CPU torch.Generator/F32 torch.randn。audio 先消耗一次完整 resolution-dependent video draw，再抽音频；video 使用独立 request substream。不要改成 MPS Generator。
- `runner.py:27,302–305` 的 video seed 为 `(request_seed + request_index*1_000_003) % 2**63`；audio 来自 teacher artifact 明确序列。
- `denoise_schedule.py:7,15–20` CPU F32 linspace、time-shift、unique_consecutive；Stage3 取 `(0,16,33,49)`，四个状态是三次 noisy forward。video shift12，audio shift3。
- `model/denoise.py:24–37,291–321,332–351` state/velocity/update 为 F32，保留运算顺序，teacher callback 在 update 之后。不要代换 sampler 公式以“简化”融合。

**Base10 teacher**

- `teacher.py:303–380` 是 base BF16，无 LoRA、无 W8A8 的音频-only teacher；10 states/9 forwards，捕获 state3/6/9。
- 后续请求冻结上一请求 clean audio 每声道最后40latents，reference时刻=1，不进入 target update；geometry 为 canonical target（198/199audio latents），不是固定每次207。`teacher.py:263–273,295–299,324–343` 与 `denoise.py:95–104,215–237`。
- teacher artifact `inference/base10_teacher.py:84–120,181–208` 校验完成标记、BF16、策略、prompt/seed/geometry、F32 milestone shape。不能为了 MPS 绕过这些检查。
- `streaming/runtime.py:326–357` 为当前 phase 从 request-wide teacher milestone按声道切片，并在每步后覆盖 audio。`739–743` 要求最终 published audio 与 final teacher latent `torch.equal`。
- 如复用同一 DiT 实例节省内存，teacher 时必须可靠禁用 LoRA，并隔离/清理 timestep/adapter caches；原官方独立 worker 天然提供这个隔离，移植不能丢。

**Streaming / cache / attention**

- `streaming/geometry.py:173–200` 四 phase groups=(2,2,2,1)，首请求124 native frames/37video latents/207audio。
- `geometry.py:30–84` 后续请求不重新生成5frame affine prefix：119native frames/35video latents，audio是全局40Hz half-even边界得到198或199，不可固定/四舍五入替代。
- `streaming/runtime.py:166–245` 保留 absolute RoPE与首次 media_time_origin，当前prompt右对齐；不能每request重置时间坐标。
- `attention_hook.py:190–211` condition只能看condition；media看condition+persistent cleanAV+current media，三个部分拼接后同一softmax，不能分别attention后相加。
- `cache.py:155–204` 只提交clean AV、不缓存text/padding；需50层齐全才能commit。`206–219` 保留第一个chunk的视频sink和最近两个完整AVchunk。
- `config.py:29`、`runtime.py:584–588` 音频KV每12个request重置，保留video；不是每request重置。
- `runtime.py:451–462` 持续以第一个stream chunk的逐通道mean/std做affine prefix renorm；clean commit必须使用renorm后的latent（`675–699`）。
- `runtime.py:748–790` transport prefix（video2latents、audio最多9）及canonical offset不能删。`runner.py:145–168` 发布前去transport prefix、拼接整条latent timeline，再一次VAE解码；不是每request单独decode后拼MP4。

**Text / precision**

- `model/presentation.py:12–19` plain prompt tokenizer、add_special_tokens=False，不应用 chat template/BOS/EOS。
- `model/text_encoder.py:23–25,195–214,223–283,298–306` 只用embedding+前50language layers，返回未final-norm的BF16 hidden，忽略vision tower/final norm/后续language layers。
- `model/dit.py:35–51` 的12个FP32参数和rope.inv_freq必须保留。不能全模型 `.bfloat16()`；本机 checkpoint 头确认为13个F32 tensor、522个BF16 tensor。
- `model/layers.py:696–741` final video/audio projection是FP32且返回FP32 velocity；这次直接移植应自然保留，不能沿用早先 h3.c 的BF16 final优化。
- `inference/lora_checkpoint.py:261–283` 将416个F32 adapter tensor转BF16计算，208targets/rank128/alpha128；MPS BF16 adapter是官方语义。TP1与TP2归约顺序不同，所以跨设备不能承诺bitwise，但要用同输入fixture量化差异。

## Bundled VAE：新增确认的 MPS 精度风险

固定 HF 源目录为 `/Users/supreme-paca/Projects/h3-metal/MiniMax-H3/FL2VA/video_vae/`。

1. **`vae_vit.py:304–305,356–357`** 用 `torch.autocast('cuda', enabled=False)` 保护 x_embedder 与 proj_out 的模块dtype线性计算。外层若只把 `output.py:778–782` 改成 `autocast('mps',float16)`，内层 CUDA disable 不会关闭 MPS autocast，FP32 输入/权重仍可被 MPS autocast降精度。`base_module.py:183–196` RoPE也有同样写死的保护域。应在后端适配层保持相应实际device的禁用域；保护官方下载文件原样与hash，可在repo适配副本/狭窄包装中修改。
2. 该问题已用 torch2.14.1 **仅检查 autocast flags、没有tensor/GPU操作**确认：`/private/tmp/taomate-mps-audit/autocast_context.py`、`autocast-context.json`。outer MPS=true，inside cuda-disabled MPS仍=true。
3. 24个固定组件源没有 `prepare_decoder_autocast_weights` 定义；进一步导入组件类并检查继承链，wrapper 与 source class 都没有该属性（`vae-prepare-method.json`，只导入类，没有模型构造、权重加载或GPU）。official `model/output.py:757–759` 是可选 getattr guarded call，在此bundle为no-op，不应假设会预转权重。原始F32权重配合CUDA autocast，是混合FP16执行，不是wholemodel.half。
4. `base_module.py:45–48` 与 `attention.py:22–26` 默认 `MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM=1`；norm输入F32然后恢复hidden dtype。需保留F32 norms、残差与输出例外，不要粗暴全VAE FP16。
5. **`flash.py:55–72`** 本身是纯torch SDPA，NHD转BHND，无FA4/CUTLASS依赖。`MINIMAX_H3_TORCH_SDPA_BACKEND` 默认auto；若用户环境强制flash/cudnn/efficient，MPS可能不兼容，必须记录/验证该环境值。该函数结尾 `nan_to_num(0)` 是原代码行为，会掩盖中间NaN，tiny验收宜在hook处检查SDPA前后finite，而非只看最终帧。
6. `flash.py:75–85` 的masked路径分配dense B×H×Q×KV bool mask；但当前source/config.json为causal_decoder=false，`klvae.py:1188` 会传给decoder，因此标准生成没有block-causal mask。不要把潜在masked路径当当前必经OOM。
7. `klvae.py:1218–1223` 解码实际是post_quant_conv **1×1×1 Conv3d** +36层ViT；CNN encoder复杂3D卷积不在T2VA解码主路。MPS tiny算子应覆盖该1×1×1 Conv3d、F32 RMSNorm、FP16/BF16 SDPA、noncontiguous QKV。
8. `klvae.py:278–288` stack_tiling=True会把本进程所有空间tile堆成一个batch。官方多卡每rank分到较少tile，单卡需要控制tile batch规模来约束峰值，保持tile顺序、blend与temporal overlap计算。
9. `minimax_h3_video_vae.py:106–113` 先构造F32 model再load_file/state copy；加载瞬间存在参数+CPU源tensor。9.70GiB是payload总量，不是加载峰值。
10. audio源 `dac_audio_vae.py:211–225` 为F32 dec_in_proj+BigVGAN；`dac_alias_free_resample.py:26–28` 的replicate pad、grouped conv_transpose1d及后续裁剪需要MPS小测。`output.py:932–944` 无video式autocast，保持F32，不能一起套外层half域。

## 128 GiB 单机内存下界（已读取真实头）

工具：`python3 /private/tmp/taomate-mps-audit/header_inventory.py`；结果 `header-inventory.json`。只读取8字节长度和JSON头；没有读取大tensor payload。每个张量都核对 shape×dtype bytes 与 data_offsets。

| 组件 | tensor数 | payload bytes | GiB |
| --- | ---: | ---: | ---: |
| 完整DiT | 535 | 66,280,430,144 | 61.72846 |
| 官方选中text embedding+50层 | 551 | 50,315,658,240 | 46.86011 |
| video VAE全部F32 | 560 | 10,415,484,128 | 9.70018 |
| audio VAE全部F32 | 1087 | 605,306,340 | 0.56374 |
| adapter checkpoint F32 | 416 | 2,480,930,816 | 2.31055 |
| adapter runtime BF16（算术） | 416 | 1,240,465,408 | 1.15527 |

DiT+text本身108.58857GiB；DiT+BF16 LoRA为62.88373GiB。全DiT转F32后权重123.39276GiB，不能当作MPS不支持某算子的全模型fallback。最大单tensor是text embedding 1,555,824,640B，DiT最大AdaLN weight 520,224,768B。

`pipeline.py:70–83` 先加载text再加载DiT；`_offload_text:101–104` 仅搬CPU，Apple统一内存中仍占46.86GiB。适配需预计算所有prompt embedding后真正释放text，再加载DiT；DiT阶段完成后释放DiT/KV/adapter再加载VAE。搬CPU不等于释放统一内存。

480p每video latent为405行；50层、K/V、56heads×128×BF16，每history token全层合计1,433,600B。根据官方12request phase/retention源码模拟：最大retained17.60788GiB，commit前后完整cache约23.16475GiB；`cache.py:273–282` 裁剪时旧_history和新next_history并存，可达40.77263GiB。上述不含temporary activations/allocator/SDPA workspace。逐层裁剪释放旧tensor可降低此峰值，但必须保持row selection、KV ownership、异常后不可复用半更新cache等行为。

另有backend相关attention峰值未知：如SDPA backend实际显式形成56×Q×KV分数矩阵，4k query/17k KV的FP32矩阵就达十几GiB；这不是已证实的MPS分配行为，需真实形状SDPA小模型测量。必要时仅切query batch、每query仍看全部KV，保持同一softmax域，不能切KV后独立softmax求和。

VAE子段头统计在 `vae-weight-sections.json`：video decoder9.02836GiB、encoder0.67180GiB；audio decoder+dec_in_proj约0.24192GiB，其他权重主要encoder。先保证完整官方加载/解码一致性，再考虑只加载decoder的优化。

## 最小独立验收建议

1. 在无CUDA/triton/vllm/FA3的环境，import全部MPS可达模块，CLI help/config/preflight均通过；TP1/Ulysses1 synthetic linear和collectives无需dist init，FP32 exceptions逐一断言。
2. CPU生成固定噪声和schedule fixture，比较原官方/适配版实际tensor；至少首请求+continuation，验证audio先消耗video draw、teacher40latent冻结和state3/6/9注入顺序。
3. 小CPU↔MPS算子：BF16 linear/RMSNorm/rotary/gate/SwiGLU，F32 final head，非连续BF16 SDPA，长方形Q≠KV与condition/media可见性；不用整模型随机seed替代输入fixture。
4. cache synthetic50层测试原子commit/缺层rollback/retention/reset@request12、shape/adapter切换无旧cache、MPS device contract。比较内存优化前后所有保留KV逐元素一致。
5. tiny VAE组件实际forward，记录各敏感模块输入/weight/output dtype，确认外层MPS autocast不会吞掉FP32投影/Rotary保护；audio只F32。再真实单请求AV，随后双请求continuation。
6. 环境receipt记录 Torch/macOS、fast_math/fallback/prefer_metal 与VAE相关环境值。默认不应以全局CPU fallback掩盖不支持算子；fallback若使用需明确到算子和时间。PyTorch官方 [MPS环境变量](https://docs.pytorch.org/docs/stable/mps_environment_variables.html) 分别说明 `PYTORCH_MPS_FAST_MATH`、`PYTORCH_ENABLE_MPS_FALLBACK`、`PYTORCH_MPS_PREFER_METAL`。
7. 计时必须在MPS synchronize前后；内存同时记录进程RSS/footprint/swap增量与 `torch.mps.current_allocated_memory`、`driver_allocated_memory`。后者包含driver/framework缓存，前者只tensor；都不是自动的整个进程生命周期峰值。[PyTorch MPS API](https://docs.pytorch.org/docs/stable/mps.html)。

本审计的动态证据只有头解析、autocast状态检查及组件类属性检查；未声称全模型MPS已运行、未声称官方CUDA与MPS数值parity，也未把早先native C分钟样本作为这次官方Python适配结果。父任务另行完成的真实VAE tiny MPS验收以新repo的 `docs/verification/component-smoke.json` 为准，不属于本审计执行内容。
