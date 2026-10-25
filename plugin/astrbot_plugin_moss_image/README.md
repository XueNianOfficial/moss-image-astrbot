# MOSS-Image 绘图插件

让 AstrBot 调用本机 **MOSS-Image**（由 ai-toolkit 推理引擎驱动：int8 convrot8 量化 + 层卸载，LoRA 以 hook 动态挂载、不合并权重）出图，带
**Prompt Enhancer** 与 **Layout Enhancer**，并注册 LLM 工具让 bot 能自主画图。

## 依赖

- 本机 MOSS-Image 服务（本仓库的服务端），默认监听 `http://127.0.0.1:8000`
  （装过开机自启会自动拉起；手动启动服务端目录下的 `run.ps1` / `run.bat`）
- 无额外 Python 依赖（仅用 AstrBot 自带的 aiohttp）

## 命令

| 命令 | 说明 |
| --- | --- |
| `/draw <描述>` | 出图，别名 `/绘图` `/画图` `/画画` `/moss` `/生图` |
| `/drawhelp` | 用法说明 |
| `/drawstatus` | 服务状态、已加载模型、显存、空闲倒计时 |
| `/loras` | 查看 LoRA 目录（名称/权重/描述/启用状态） |
| `/drawunload` | 立即卸载模型释放显存 |
| `/enhance <描述>` | 只预览增强后的提示词，不画图 |
| `/layout <描述>` | 只预览编译好的 Layout 提示词，不画图 |

### `/draw` 参数

```
/draw 雨夜的霓虹街道，一只白狐回头 --size 1024x1536 --steps 30 --seed 42 --no blurry, 水印
```

| 参数 | 说明 |
| --- | --- |
| `--size WxH` | 像素尺寸，必须是 16 的倍数，64–2048 |
| `--ar 16:9` | 按比例取推荐尺寸（基于 1024² 面积） |
| `--steps N` | 采样步数 |
| `--guidance G` | 引导强度（CFG） |
| `--seed N` | 复现同一张图 |
| `--n N` | 一次出几张（默认上限 2） |
| `--no <文本>` | 负面提示词 |
| `--lora 名称:权重` | 指定 LoRA（名称来自配置里的 LoRA 目录），多个用逗号 |
| `--enhance` | 强制增强提示词 |
| `--raw` | 完全按原文出图 |
| `--layout` | 走 Layout Enhancer |

## LoRA 目录

LoRA 不是写死的，而是配置里一段**纯文本**（`lora.catalog_lines`），每行一条：

```
名称 | 文件路径 | on/off | 权重 | 描述 | 触发词
```

```
# 注释行会被忽略
kemono | G:/ai-toolkit/output/my_kemono_lora/my_kemono_lora.safetensors | on | 0.7 | 兽人/Kemono 画风：柔和赛璐璐上色、蓬松毛发；适合兽设、兽人插画。 |
mychar | G:/loras/mychar.safetensors | off | 1.0 | 角色 LoRA「某角色」：白色毛发、红眼、冰蓝渐变耳与尾；画这个角色本人时使用。 | 角色触发词
```

| 字段 | 说明 |
| --- | --- |
| 名称 | 引用名，请求 / 命令里用它，例如 `kemono` |
| 文件路径 | `.safetensors`（diffusers / ai-toolkit / kohya / lokr 都能解析） |
| on/off | **默认是否启用**（off 的条目仍会以权重 0 挂载，随时可被请求唤醒） |
| 权重 | **默认强度** |
| 描述 | AI 依据它判断该不该用、用多重；写得越具体越准 |
| 触发词 | 启用时自动追加到提示词，多个用逗号，可留空 |

多条可以写在不同行，也可以在同一行用 `;` 分隔（输入框只有一行时用分号）。

机制：

1. 插件启动时把目录里**所有**条目注册到服务端（`POST /v1/loras/load`，只写注册表，不会加载模型）；
2. 每次出图都下发一份完整的 `{名称: 权重}`：
   - on 的用默认权重，off 的发 0（ai-toolkit 的 hook 在 0 时直接短路，不耗算力）；
   - 同一集合只改强度，**不会重新读取 LoRA 文件**；
3. 目录摘要（名称 + 描述 + 默认权重 + 启用状态 + 触发词）会注入系统提示，
   `lora.allow_bot_choice` 开启时 AI 会自己决定用哪个、用多重，并通过
   `moss_draw_image` 的 `lora` 参数传下来；
4. 用户在命令里写的 `--lora` 优先于 AI 的判断；
5. 服务端重启会丢注册表 —— 插件遇到 `LoRA not registered` 会自动补注册并重试一次。

`/loras` 可以随时查看当前目录（✅ 启用 / ▫️ 关闭 / ⚠️ 文件缺失）。

## 采样参数

`generation` 段里这些都是可改的（面板或直接改 JSON）：

| 配置 | 说明 |
| --- | --- |
| `steps` | 采样步数（默认 30） |
| `guidance` | CFG 强度（默认 3.0；1.0 关闭 CFG，速度快一倍） |
| `negative_prompt` | **默认负面提示词**，每次出图都带，命令里的 `--no` 追加在后面 |
| `shift` | flow-matching sigma shift；`0` = 跟随服务端（官方 5.0），调低更贴近训练分布 |
| `uncond_skip_layers` | CFG 跳过的 DiT 层，默认 `22`；`off` = 不跳过（速度减半）；多个用逗号 |

后三项通过请求体传给服务端，服务端直接改 ai-toolkit holder 上的
`sample_shift` / `uncond_skip_layers`，**不重新加载模型**。

## Prompt Enhancer

短于 `enhance.auto_enhance_below`（默认 60 字符）的描述会自动扩写。规则取自
官方 `moss-image-prompt-enhancer` skill：

- 只描述可见内容：主体与动作、外观、场景、镜头、材质、光线、色调、氛围
- 保留用户的主体、数量、关系、风格与需要出现在图里的文字（原文照抄）
- 不堆 "masterpiece, best quality" 之类的质量词，不写流程说明
- 默认英文散文体，60–140 词

LLM 不可用或超时会**自动回退到原文**，不会因为增强失败而画不出图。

## Layout Enhancer

适合海报、封面、带标题或需要精确摆放元素的画面。流程严格按官方 skill：

1. LLM 只负责写 **draft JSON**：`width/height`、`bbox_coord_space="abs"`、
   `bbox_format="xyxy"`、`summary`、扁平的 `items`（`object`/`character`/`text`）
2. 坐标换算、越界校验、中心化 `yxyx` 压缩全部交给内置的官方
   `compile_prompt.py`，LLM 不碰坐标数学
3. 校验失败会把具体错误回传给 LLM 重写（默认重试 2 次），仍失败则退化为普通描述

画布尺寸以命令/配置为准，`compile_prompt` 会拒绝超出画布的框。

## LLM 工具

注册了 `moss_draw_image(prompt, size, steps, seed, negative_prompt, enhance, layout, lora)`。
启用了工具调用的模型在用户说「画一张…」「出图」「来一张」时会直接调用它，
图片作为副作用发送，函数只返回一句状态，避免模型重复描述画面。

`lora` 参数（如 `xuenian:0.9,kemono:0.5`）配合注入的目录摘要，让模型自主决定
「这次要不要挂 LoRA、挂哪条、多重」；不匹配时留空即可。

配置里 `behaviour.inject_tool_guidance` 会在系统提示中补一句「你可以直接画图」，
对付一些人设里「不要调用工具 / 直接回答」的设定。

## 结构

```
astrbot_plugin_moss_image/
├── main.py             # 命令、LLM 工具、结果组装
├── moss_client.py      # 与本机服务的 HTTP 客户端
├── prompt_enhancer.py  # Prompt Enhancer
├── layout_enhancer.py  # Layout Enhancer（LLM draft + 校验重试）
├── prompt_parser.py    # --flag 解析、尺寸换算
├── compile_prompt.py   # 官方 Layout 编译器（原样内置）
└── _conf_schema.json   # 插件配置面板
```

## 常见问题

**「无法连接 MOSS-Image 服务」**
服务没起来。运行 `G:\MOSS-Image-Server\run.ps1`，或用计划任务 `MOSS-Image Server` 启动；
`/drawstatus` 可以看到连接状态。

**第一次出图特别慢**
服务是懒加载的：第一个请求会先加载模型（约 30 秒），空闲 10 分钟后自动卸载，
下一个请求再加载。想常驻可以调大服务端 `--idle-unload`。

**出图很慢 / 显存紧张**
训练任务在占用显卡时，文本编码器会留在内存，单次编码约 10 秒。等显卡空闲后重启服务会更块。

**Layout 总是失败**
换个更明确的描述，或把 `layout.max_retries` 调大。Layout 对文字框尺寸敏感，
标题太长时更适合放大画布（`--size 1536x1024`）。
