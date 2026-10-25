# MOSS-Image 本地推理服务 + AstrBot 绘图插件

[![ci](https://github.com/XueNianOfficial/moss-image-astrbot/actions/workflows/ci.yml/badge.svg)](https://github.com/XueNianOfficial/moss-image-astrbot/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/XueNianOfficial/moss-image-astrbot?include_prereleases)](https://github.com/XueNianOfficial/moss-image-astrbot/releases)
[![license](https://img.shields.io/github/license/XueNianOfficial/moss-image-astrbot)](LICENSE)

在**消费级显卡（12 GiB 起）**上跑 [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
的本地 HTTP 服务，并附带一个让 AstrBot 能自主画图的插件：

- **低显存**：int8 `convrot8` 量化 + 分层卸载，实测 1024² 峰值 12.6 GiB、空闲卸载后 2.7 GiB
- **不合并权重**：LoRA 通过 forward hook 动态挂载，换 LoRA / 改强度不重载模型
- **空闲自动卸载**：默认 10 分钟没有请求就把显存还回去，下个请求再懒加载
- **Prompt Enhancer**：把一句话扩写成 MOSS-Image 官方风格的长提示词
- **Layout Enhancer**：海报/带文字排版，LLM 只写草稿框，坐标换算走官方 `compile_prompt`
- **LoRA 目录**：配置里带描述，AI 自己判断该不该挂、挂多重

<p align="center">
  <img src="assets/sample-kemono-1024.png" width="45%" alt="sample: kemono LoRA @0.7" />
  <img src="assets/sample-lora-768.png" width="45%" alt="sample: character LoRA @1.0" />
</p>

> 左：风格 LoRA @0.7，1024²/20 步，17 s ｜ 右：角色 LoRA @1.0，768²/12 步，7 s

## 工作原理

```
AstrBot ──/draw──► 插件（Prompt/Layout 增强、LoRA 选择）
                        │  HTTP /v1/moss/generate
                        ▼
              MOSS-Image 服务（FastAPI）
                        │  懒加载 / 空闲卸载 / LoRA 注册表
                        ▼
              ai-toolkit 推理引擎
        （convrot8 int8、层卸载、hook LoRA）
                        │
                        ▼
        MOSS-Image-SFT（diffusers 布局）+ 你的 LoRA
```

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

常用参数：

```powershell
.\install.ps1 -Autostart                        # 顺便设置开机自启
.\install.ps1 -ModelDir D:\models\MOSS-Image-SFT -SkipModel   # 已有模型
.\install.ps1 -SkipPlugin                       # 只装服务端
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
```

## 快速开始

```powershell
.\run.ps1                     # 默认 http://127.0.0.1:8000
python scripts\smoke_test.py  # 懒加载、OpenAI 接口、SSE、卸载/重载 全套自检
```

首次请求会加载模型（约 20 s），之后 1024²/20 步约 17 s。10 分钟没请求会自动卸载显存。

```bash
curl -X POST http://127.0.0.1:8000/v1/moss/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a white wolf beastman, kemono style","size":"1024x1024","steps":20}'
```

### 接入 AstrBot

1. 安装脚本已把插件放到 `%USERPROFILE%\.astrbot\data\plugins\astrbot_plugin_moss_image`
2. 打开 AstrBot WebUI → 插件管理 → **重载**
3. 直接说「画一张雨夜霓虹街道的白狐」，或在群里：

```
/draw 雨夜的霓虹街道，一只白狐回头 --size 1024x1536 --steps 30
/loras              # 查看 LoRA 目录
/drawstatus         # 服务状态、模型、显存
```

插件注册了 LLM 工具 `moss_draw_image`，支持工具调用的模型会自己决定何时画图、用哪个 LoRA。

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

插件配置（WebUI 里可直接改）：负面提示词、步数、CFG、shift、CFG 跳过层、LoRA 目录文本、
是否允许 AI 自选 LoRA、冷却与权限等。LoRA 目录每行一条：

```
名称 | 文件路径 | on/off | 权重 | 描述 | 触发词
```

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/moss/generate` | 原生接口（推荐） |
| `POST` | `/v1/images/generations` | OpenAI 兼容 |
| `POST` | `/v1/moss/generate/stream` | SSE 进度流 |
| `GET` | `/v1/status` | 状态、LoRA、显存、统计 |
| `POST` | `/v1/loras/load` | 注册 LoRA（不加载模型） |
| `POST` | `/v1/loras/{name}/scale` | 改全局强度 |
| `DELETE` | `/v1/loras/{name}` | 移除 |
| `POST` | `/v1/admin/unload` / `/load` | 手动释放/预加载 |

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

## 显存与性能（RTX 5090D 32 GiB，实测）

| 场景 | 数值 |
| --- | --- |
| 1024² / 20 步 | 17 s（峰值 12.6 GiB） |
| 768² / 12 步 | 7 s |
| 模型加载 | ~20 s（懒加载，首次请求） |
| 空闲卸载后显存 | 2.7 GiB |

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
moss_server/        推理服务（FastAPI + 引擎封装 + LoRA 注册表 + 校验）
plugin/             AstrBot 插件（命令、LLM 工具、增强器、LoRA 目录）
scripts/            冒烟测试、压测、画质检查
tests/              单元测试 + 插件集成测试（无需 GPU 的部分可直接跑）
install.ps1|.sh     一键安装
config.example.toml 配置模板
```

## 开发与测试

```bash
python tests/test_schemas.py          # 请求/配置校验（无需 GPU）
python tests/test_lora.py             # LoRA 注册表 + 头扫描
python tests/test_plugin_helpers.py   # 插件纯函数（解析、提示词、布局）
python tests/test_plugin_integration.py   # 需要服务端 + AstrBot
python scripts/smoke_test.py          # 端到端
```

## 常见问题

见 [docs/faq.md](docs/faq.md)：显存不够、下载中断、插件不支持某些字段、Windows 编码等。

## 许可证

代码 MIT（见 [LICENSE](LICENSE)）。模型权重 Apache-2.0，ai-toolkit MIT，AstrBot AGPL-3.0。
如果分发时包含 AstrBot 插件，请遵守 AGPL 的相应要求。

## 致谢

- [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS）——模型、prompt/layout skill
- [ai-toolkit](https://github.com/ostris/ai-toolkit)（ostris）——推理引擎、量化、LoRA 加载
- [AstrBot](https://github.com/AstrBotDevs/AstrBot)——插件宿主
