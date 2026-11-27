# MOSS-Image 本地推理服务 + AstrBot 绘图插件

[![ci](https://github.com/XueNianOfficial/moss-image-astrbot/actions/workflows/ci.yml/badge.svg)](https://github.com/XueNianOfficial/moss-image-astrbot/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/XueNianOfficial/moss-image-astrbot?include_prereleases)](https://github.com/XueNianOfficial/moss-image-astrbot/releases)
[![license](https://img.shields.io/github/license/XueNianOfficial/moss-image-astrbot)](LICENSE)

在**消费级显卡（12 GiB 起）**上跑 [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
的本地 HTTP 服务，并附带一个让 AstrBot 能自主画图、改图的插件：

- **低显存**：int8 `convrot8` 量化 + 分层卸载，实测 1024² 峰值 12.6 GiB、空闲卸载后 2.7 GiB
- **不合并权重**：LoRA 通过 forward hook 动态挂载，换 LoRA / 改强度不重载模型
- **空闲自动卸载**：默认 10 分钟没有请求就把显存还回去，下个请求再懒加载
- **图像参考编辑**：同一个 ai-toolkit 引擎加载 [Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit)
  的编辑 LoRA，按文字指令换背景 / 换装 / 换光线，支持 1-4 张参考图；参考图以干净 latent 进入，
  K/V 缓存跨采样步复用
- **提示词交给独立 skill**：[moss-image-prompt-skill](https://github.com/Moeblack/moss-image-prompt-skill)
  运行时从磁盘读取，插件不内置任何提示词规则（也不打包其内容）
- **Layout Enhancer**：海报/带文字排版，LLM 只写草稿框，坐标换算走官方 `compile_prompt`
- **LoRA 目录**：配置里带描述，AI 自己判断该不该挂、挂多重

<p align="center">
  <img src="assets/sample-kemono-1024.png" width="45%" alt="sample: kemono LoRA @0.7" />
  <img src="assets/sample-lora-768.png" width="45%" alt="sample: character LoRA @1.0" />
</p>

> 左：风格 LoRA @0.7，1024²/20 步，17 s ｜ 右：角色 LoRA @1.0，768²/12 步，7 s
>
> 改图示例：同一张图「把背景换成晴天的海边，保留角色」→ 1024²/12 步，模型热态 **13 s**

## 工作原理

```
AstrBot ──/draw───► 插件（skill 写提示词、Layout、LoRA 选择）
        ──/edit───►        │
                            │ HTTP /v1/moss/generate   /v1/moss/edit
                            ▼
                  MOSS-Image 服务（FastAPI）
                            │  懒加载 / 空闲卸载 / LoRA 注册表
                            ▼
              ai-toolkit 推理引擎（MOSS-Image 模型）
        ┌───────────────────┴────────────────────┐
        │  prompt       → 文本 token + 目标 latent │  文生图
        │  prompt+参考图 → 文本 token + 参考 latent │  改图（t=0 调制，
        └────────────────────────────────────────┘   参考 K/V 缓存）
                            │
                            ▼
        MOSS-Image-SFT（convrot8 int8）+ 你的 LoRA / 编辑 LoRA
```

改图不是另一套模型：ai-toolkit 的 MOSS 实现本身就有参考图分支（参考 token 以 t=0
调制、只与自身互相可见、K/V 可跨步缓存）。服务端在**同一个已加载的模型**上翻转三个开关
（`is_edit` / `kv_cache` / 视觉提示词编码）并临时挂上编辑 LoRA，请求结束再翻回来。
所以「画图 → 改图 → 画图」全程**不重载权重**：模型只加载一次、只占一份显存。

模型加载、量化、采样、LoRA 解析全部由 [ai-toolkit](https://github.com/ostris/ai-toolkit)
的推理引擎完成——本仓库只做服务化、调度和插件，所以能跟着 ai-toolkit 一起升级。

## 环境要求

| 项 | 要求 |
| --- | --- |
| 系统 | Windows 10/11（一键脚本）；Linux 也可（`install.sh`） |
| 显卡 | NVIDIA，**≥ 12 GiB 显存**（默认设置）；8 GiB 可把 `layer_offloading_transformer_percent` 调高 |
| 驱动 | 支持 CUDA 12.8 的新驱动（RTX 50 系需要） |
| Python | 3.10 – 3.13 |
| 磁盘 | 模型约 30 GB + 依赖约 8 GB |
| 其他 | git；AstrBot 4.16+（可选，只影响插件） |

## 一键安装

### Windows

```powershell
git clone https://github.com/XueNianOfficial/moss-image-astrbot.git
cd moss-image-astrbot
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

脚本会依次：检查驱动/GPU → 克隆 ai-toolkit（钉在测试过的 commit）→ 建 venv 装 CUDA torch 与依赖
→ 从 Hugging Face 下载模型 → 生成 `config.toml` → 安装 AstrBot 插件 →（可选）注册开机自启。

默认还会装好**改图**与**提示词 skill** 两个可选件（下载约 300 MB 的编辑 LoRA、克隆
编辑 LoRA 与 moss-image-prompt-skill）。不想要可以加 `-SkipEdit` / `-SkipSkill`。

常用参数：

```powershell
.\install.ps1 -Autostart                        # 顺便设置开机自启
.\install.ps1 -ModelDir D:\models\MOSS-Image-SFT -SkipModel   # 已有模型
.\install.ps1 -SkipPlugin                       # 只装服务端
.\install.ps1 -SkipEdit                         # 不装改图（少 300 MB）
.\install.ps1 -Cpu                              # 只验证接线（CPU 推理极慢）
```

### Linux / macOS

```bash
git clone https://github.com/XueNianOfficial/moss-image-astrbot.git
cd moss-image-astrbot
./install.sh                     # MODEL_DIR=/data/MOSS-Image-SFT ./install.sh
```

### 手动安装（不想用脚本）

```bash
git clone https://github.com/ostris/ai-toolkit ../ai-toolkit
python -m venv ../ai-toolkit/.venv
../ai-toolkit/.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
../ai-toolkit/.venv/Scripts/pip install -r ../ai-toolkit/requirements.txt -r requirements.txt
hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir D:/models/MOSS-Image-SFT
cp config.example.toml config.toml   # 改 aitk_root / model_dir

# 可选：改图（只需要编辑 LoRA，运行时融合成 ai-toolkit 的 qkv 布局）
hf download chinoll/Moss-Image-Edit --local-dir .moss-image-edit-adapter
# 然后在 config.toml 里设置 edit_adapter

# 可选：提示词 skill（CC BY-NC-ND 4.0，不打包进本仓库）
git clone https://github.com/Moeblack/moss-image-prompt-skill \
  ~/.astrbot/data/skills/moss-image-prompt-skill
```

## 快速开始

```powershell
.\run.ps1                     # 默认 http://127.0.0.1:8000
python scripts\smoke_test.py  # 懒加载、OpenAI 接口、SSE、卸载/重载 全套自检
```

首次请求会加载模型（约 20 s），之后 1024²/20 步约 17 s。10 分钟没请求会自动卸载显存。

```bash
# 文生图
curl -X POST http://127.0.0.1:8000/v1/moss/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a white wolf beastman, kemono style","size":"1024x1024","steps":20}'

# 改图（参考图 base64 放进 images；也可用 data:image/png;base64,...）
curl -X POST http://127.0.0.1:8000/v1/moss/edit \
  -H "Content-Type: application/json" \
  -d @edit.json     # {"prompt":"把背景换成晴天的海边，保留角色","images":["<b64>"],"steps":12}
```

### 接入 AstrBot

1. 安装脚本已把插件放到 `%USERPROFILE%\.astrbot\data\plugins\astrbot_plugin_moss_image`
2. 打开 AstrBot WebUI → 插件管理 → **重载**
3. 直接说「画一张雨夜霓虹街道的白狐」，或在群里：

```
/draw 雨夜的霓虹街道，一只白狐回头 --size 1024x1536 --steps 30
/edit 把背景换成晴天的海边，保留人物        # 图片和指令一起发，或引用一张图
/loras              # 查看 LoRA 目录
/drawstatus         # 服务状态、模型、显存、skill、改图状态
```

插件注册了 LLM 工具 `moss_draw_image` 与 `moss_edit_image`，支持工具调用的模型会自己
决定何时画图 / 改图、用哪个 LoRA，并在写提示词前先读 prompt skill。

## 配置

服务端 `config.toml`（完整说明见 [docs/configuration.md](docs/configuration.md)）：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `aitk_root` | — | ai-toolkit 目录（提供推理引擎） |
| `model_dir` | — | MOSS-Image diffusers 模型目录 |
| `quantize` / `qtype` | `true` / `convrot8` | int8 权重量化 |
| `low_vram` | `true` | 组件常驻内存，用时再上卡 |
| `layer_offloading_transformer_percent` | `0.0` | 0=常驻，1=全卸载（更省显存更慢） |
| `layer_offloading_text_encoder_percent` | `1.0` | 文本编码器卸载比例 |
| `idle_unload_seconds` | `600` | 空闲多久卸载；`0` 关闭 |
| `default_steps` / `default_guidance_scale` | `30` / `3.0` | 生成默认值 |
| `default_shift` | `5.0` | flow-matching shift |
| `default_uncond_skip_layers` | `[22]` | CFG 跳过的 DiT 层 |
| `loras` | `[]` | 启动时注册的 LoRA（插件会自己注册，通常留空） |
| `lora_mode` | `hook` | `hook`=动态挂载，`merge`=写入权重 |
| `edit_adapter` | `""` | 编辑 LoRA 目录（`adapter_config.json` + `adapter_model.safetensors`） |
| `edit_steps` / `edit_guidance_scale` | `20` / `4.0` | 改图默认采样参数 |
| `edit_max_ref_images` / `edit_max_edge` | `4` / `1024` | 参考图数量与长边上限 |
| `edit_uncond_skip_layers` | `[]` | 改图的 CFG 跳过层（编辑 LoRA 训练时未跳过） |

插件配置（WebUI 里可直接改）：负面提示词、步数、CFG、shift、CFG 跳过层、LoRA 目录文本、
prompt skill 路径与章节、改图开关、冷却与权限等。LoRA 目录每行一条：

```
名称 | 文件路径 | on/off | 权重 | 描述 | 触发词
```

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/moss/generate` | 原生文生图（推荐） |
| `POST` | `/v1/moss/edit` | **图像参考编辑**（参考图 + 指令） |
| `POST` | `/v1/images/generations` | OpenAI 兼容 |
| `POST` | `/v1/moss/generate?stream=true` | SSE 进度流 |
| `GET` | `/v1/status` | 状态、LoRA、显存、统计、改图状态 |
| `GET` | `/v1/edit/status` | 改图可用性与模型状态 |
| `POST` | `/v1/loras/load` | 注册 LoRA（不加载模型） |
| `POST` | `/v1/loras/{name}/scale` | 改全局强度 |
| `DELETE` | `/v1/loras/{name}` | 移除 |
| `POST` | `/v1/admin/unload` / `/load` | 手动释放/预加载 |
| `POST` | `/v1/admin/edit/unload` | 立即卸载改图模型 |

文生图请求体：

```jsonc
{
  "prompt": "a white wolf beastman, kemono style",
  "negative_prompt": "low quality, watermark",
  "size": "1024x1024",              // 或 width/height；支持 "16:9"
  "steps": 20,
  "guidance_scale": 3.0,
  "shift": 5.0,                      // 可选
  "uncond_skip_layers": [22],        // 可选，[] 表示不跳过
  "seed": 42,
  "loras": {"my_lora": 0.7},         // 只改本次强度
  "response_format": "url"           // 或 b64_json
}
```

改图请求体：

```jsonc
{
  "prompt": "把背景换成晴天的海边，保留人物",
  "images": ["<base64 或 data:image/png;base64,...>"],   // 1-4 张，顺序有意义
  "size": "1024x1024",        // 省略则按第一张参考图比例自动取
  "steps": 12,                // 省略用 edit_steps
  "guidance_scale": 4.0,
  "shift": 5.0,
  "seed": 7,
  "max_edge": 1024,           // 参考图缩放长边
  "use_kv_cache": true,       // 参考图 KV 缓存（默认开）
  "response_format": "url"
}
```

## 显存与性能（RTX 5090D 32 GiB，实测）

| 场景 | 数值 |
| --- | --- |
| 1024² / 20 步 | 17 s（峰值 12.6 GiB） |
| 768² / 12 步 | 7 s |
| 模型加载 | ~20 s（懒加载，首次请求） |
| 空闲卸载后显存 | 2.7 GiB |
| 改图模型加载 | ~21 s（与文生图同一套 convrot8 量化） |
| 改图 1024² / 12 步（热，与文生图共用模型） | **13-15 s** |
| 文生图 ↔ 改图切换 | **0 s**（不重载权重） |

## LoRA

不合并权重：LoRA 以 hook 方式挂载，强度实时生效，同一集合改强度**不重读文件**。

```bash
# 注册（插件会自动做）
curl -X POST http://127.0.0.1:8000/v1/loras/load -H "Content-Type: application/json" \
  -d '{"path":"G:/loras/mylora.safetensors","name":"mylora","scale":0.8}'
```

支持 diffusers / ai-toolkit / kohya / LoKr 格式。

## 目录结构

```
moss_server/        推理服务（FastAPI + 引擎封装 + LoRA 注册表 + 改图 + 校验）
  edit.py           改图服务：请求编排、参考图落盘、状态上报（模型与文生图共用）
  edit_lora.py      把 Moss-Image-Edit 适配器融合成 ai-toolkit 的 qkv 布局（带缓存）
  engine.py         MossEngine：文生图与参考图编辑共用一个加载好的模型
plugin/             AstrBot 插件（命令、两个 LLM 工具、prompt skill 加载、LoRA 目录）
scripts/            冒烟测试、压测、画质检查
tests/              单元测试 + 插件集成测试（无需 GPU 的部分可直接跑）
install.ps1|.sh     一键安装
config.example.toml 配置模板
```

## 开发与测试

```bash
python tests/test_schemas.py          # 请求/配置校验（无需 GPU）
python tests/test_lora.py             # LoRA 注册表 + 头扫描
python tests/test_edit.py             # 改图请求、适配器融合、生命周期（无需 GPU）
python tests/test_prompt_skill.py     # skill 解析与输出契约（MOSS_TEST_SKILL=<SKILL.md>）
python tests/test_plugin_helpers.py   # 插件纯函数（解析、提示词、布局）
python tests/test_plugin_integration.py   # 需要服务端 + AstrBot（MOSS_TEST_EDIT=1 跑真实改图）
python scripts/smoke_test.py          # 端到端
```

## 常见问题

见 [docs/faq.md](docs/faq.md)：显存不够、改图报错、skill 找不到、下载中断、Windows 编码等。

## 许可证

代码 MIT（见 [LICENSE](LICENSE)）。

本项目**不打包**以下外部内容，均在运行时由用户自行安装：

| 组件 | 许可证 | 说明 |
| --- | --- | --- |
| MOSS-Image 权重 | Apache-2.0 | Hugging Face |
| ai-toolkit | MIT | 推理引擎，运行时导入 |
| moss-image-prompt-skill | **CC BY-NC-ND 4.0** | 提示词规则；运行时从磁盘读取，不复制、不改写、不商用 |
| AstrBot | AGPL-3.0 | 插件宿主 |

编辑 LoRA 权重（chinoll/Moss-Image-Edit）由用户自行下载；本项目只做格式转换，不打包权重，
使用该权重请遵守其发布页的授权说明。

## 致谢

- [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS）——模型、prompt/layout skill
- [ai-toolkit](https://github.com/ostris/ai-toolkit)（ostris）——推理引擎、量化、LoRA 加载
- [Moss-Image-Edit](https://github.com/Chenkin-x/Moss-Image-Edit)（Chenkin-x / chinoll）——图像编辑 LoRA 与参考图分支
- [moss-image-prompt-skill](https://github.com/Moeblack/moss-image-prompt-skill)（Moeblack）——提示词规则集
- [AstrBot](https://github.com/AstrBotDevs/AstrBot)——插件宿主
