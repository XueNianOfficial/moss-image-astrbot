# 常见问题

## 安装

**Q: `install.ps1` 报「无法加载文件，因为在此系统上禁止运行脚本」**
用 `powershell -ExecutionPolicy Bypass -File .\install.ps1` 运行，或先执行
`Set-ExecutionPolicy -Scope Process Bypass`。

**Q: 找不到 Python 3.10–3.13**
从 [python.org](https://www.python.org/downloads/) 安装 3.12 并勾选「Add python.exe to PATH」。
脚本会优先用 `py -3.12`。也可以 `-Python "C:\Python312\python.exe"` 指定。

**Q: 模型下载中断**
`hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir <目录>` 支持断点续传，重复执行即可。
国内网络可以先 `pip install -U huggingface_hub` 再设 `HF_ENDPOINT=https://hf-mirror.com`。

**Q: 没有 NVIDIA 显卡 / 显存小于 8 GiB**
这个模型不量化时约 20 GiB，量化 + 全卸载可以压到 ~10 GiB，但仍需要 NVIDIA 显卡。
CPU 推理（`-Cpu`）只适合验证接线，一张 1024² 图要几十分钟。

## 运行

**Q: 第一次请求特别慢（约 60 s）**
这是懒加载 + 预热的开销：加载 20 s、预热 20 s，之后 1024²/20 步约 17 s。
想省掉预热设 `warmup = false`（只是把开销挪到第一个真实请求）。

**Q: 显存被占满 / OOM**
按顺序调：`layer_offloading_transformer_percent` 0.3 → 0.6 → 1.0；
关掉 `warmup`；把 `idle_unload_seconds` 调小（如 120）。仍不行就降分辨率或关 `quantize=false`
以外的东西——`quantize=false` 会显著增加显存。

**Q: 服务起来了但插件连不上**
`/drawstatus` 看是否 `❌`；确认 `server.base_url` 是 `http://127.0.0.1:8000`，
服务端 `host` 是 `127.0.0.1`（不是 `0.0.0.0` 时本机也能连）。若服务端设了 `api_key`，插件里要填一样的。

**Q: 换 LoRA 需要重启吗**
不需要。插件的 LoRA 目录会在启动时注册，改强度只更新 hook，不重读文件；加新 LoRA 保存配置后重载插件即可。
服务端重启会丢注册表，插件遇到 `LoRA not registered` 会自动补注册并重试。

**Q: 请求里写了未知 LoRA 名**
返回 `400 LoRA not registered`——这是故意的，避免静默丢掉 LoRA。先 `POST /v1/loras/load` 注册。

**Q: 出图风格不对 / 不像 LoRA 角色**
看 `_build_components` 追加的参数行里 `LoRA xxx 0.8` 是否符合预期；
角色 LoRA 建议 0.9–1.0，风格 LoRA 0.5–0.8；同时确认 `trigger_words` 已填（会追加到提示词）。

## 插件 / AstrBot

**Q: `/draw` 只有「正在生成」没有图**
旧版本插件曾把 `MessageChain` 直接 `yield`，AstrBot 的发送阶段只处理 `MessageEventResult`，
图片会被静默丢弃。本仓库版本已修正为 `event.chain_result(components)`，
升级后请重载插件。若仍无图，检查 AstrBot 日志里有没有 `LoRA not registered`
或适配器上传失败。

**Q: 配置项显示不全 / 改不了**
插件配置页只渲染标准 schema 类型（string/int/float/bool/object）。
本项目刻意把 LoRA 目录做成一整段**文本**（`lora.catalog_lines`），就是为了在任何版本的
配置页里都能编辑；若面板显示异常，可直接改
`~/.astrbot/data/config/astrbot_plugin_moss_image_config.json`（UTF-8）后重载插件。

**Q: 中文提示词可以用吗**
可以。Prompt Enhancer 会把中文描述扩写成英文散文体；`--raw` 则原样发给模型（模型本身支持中文提示词）。

**Q: 群聊里怎么限制只有管理员能画**
插件配置 `behaviour.admin_only = true`，同时会限制 LLM 工具调用。

## 其他

**Q: Windows 下日志/输出中文乱码**
启动脚本已设 `PYTHONUTF8=1`。若在自建脚本里调用，请同样设置，或 `chcp 65001`。

**Q: 能商用吗**
代码 MIT；**模型权重是 Apache-2.0**，但生成内容的合规性、训练数据版权、角色 LoRA 的授权
需要你自己确认。AstrBot 是 AGPL-3.0。
