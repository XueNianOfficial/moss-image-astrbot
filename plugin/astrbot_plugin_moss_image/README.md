# MOSS-Image 绘图 / 改图插件

让 AstrBot 调用本机 **MOSS-Image**（ai-toolkit 引擎：int8 convrot8 量化 + 层卸载，LoRA 以 hook
动态挂载、不合并权重）出图，并用 **Moss-Image-Edit** 按指令改图。

提示词规则**不在插件里**：它们来自独立的
[moss-image-prompt-skill](https://github.com/Moeblack/moss-image-prompt-skill)（CC BY-NC-ND 4.0），
插件在运行时从磁盘读取，不内置、不打包。布局增强仍走官方 `compile_prompt`。

## 依赖

- 本机 MOSS-Image 服务（本仓库的服务端），默认监听 `http://127.0.0.1:8000`
  （装过开机自启会自动拉起；手动启动服务端目录下的 `run.ps1` / `run.bat`）
- 改图需要服务端配好 `edit_adapter`（编辑 LoRA），与文生图共用同一个已加载模型
- 多参考图：bot 用 `moss_edit_image` 的 `images` 参数按编号（`"1,2"`）或 URL 指定
- 改图 LoRA 单独配置：配置页「改图 LoRA」是独立清单，默认启用的条目只作用于改图
- 改图 skill：`skills/moss-image-edit/SKILL.md`（MIT），安装脚本会复制到 AstrBot 的 skills 目录
- 提示词 skill 放在 `~/.astrbot/data/skills/moss-image-prompt-skill/SKILL.md`
- 无额外 Python 依赖（仅用 AstrBot 自带的 aiohttp）

## 命令

| 命令 | 说明 |
| --- | --- |
| `/draw <描述>` | 出图，别名 `/绘图` `/画图` `/画画` `/moss` `/生图` |
| `/edit <指令>` | 改图（图与指令同发，或引用图片），别名 `/改图` `/编辑图片` `/修图` |
| `/drawhelp` | 用法说明 |
| `/drawstatus` | 服务状态、模型、显存、skill 与改图状态 |
| `/loras` | 查看 LoRA 目录（名称/权重/描述/启用状态） |
| `/drawunload` | 立即卸载模型释放显存 |
| `/enhance <描述>` | 只预览 skill 写出的提示词，不画图 |
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
| `--enhance` | 用 skill 现场改写提示词（默认不改写，交给对话模型读 skill） |
| `--mode real\|anime\|cross` | 指定写实 / 二次元 / 跨次元模式 |
| `--raw` | 完全按原文出图 |
| `--layout` | 走 Layout Enhancer |

### `/edit` 参数

```
（发一张图）/edit 把背景换成晴天的海边，保留人物 --steps 12 --seed 7 --no low quality
```

| 参数 | 说明 |
| --- | --- |
| `--size WxH` | 输出尺寸；省略则按第一张参考图比例自动决定 |
| `--steps N` | 采样步数（默认用服务端 `edit_steps`） |
| `--seed N` | 复现同一次改图 |
| `--no <文本>` | 负面提示词 |
| `--enhance` | 先用 skill 把编辑指令写细（默认原样发送，编辑指令更看重保真） |

最多同时给 `edit.max_images`（默认 4）张参考图，指令里要说明每张的用途。

## LLM 工具

| 工具 | 说明 |
| --- | --- |
| `moss_draw_image` | 出图；可带 `lora` 参数按 LoRA 描述自选 |
| `moss_edit_image` | 改图；自动取当前消息或引用消息里的图片 |

`skill.guidance` 开启时，系统提示会要求模型出图前先读 SKILL.md（按章节读，不要整篇灌入）。

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

## 提示词 skill

插件不再内置提示词规则。规则来自
[moss-image-prompt-skill](https://github.com/Moeblack/moss-image-prompt-skill)（CC BY-NC-ND 4.0），
运行时读取，两种用法：

1. **对话模型自己读**（默认）：`skill.guidance` 注入「出图前先读 SKILL.md、按章节取用」，
   模型用自己的文件工具读，再带 prompt 调工具；
2. **插件代读**（`--enhance` 或 `enhance.enabled = true`）：取 `skill.sections` 指定章节
   （默认 `0,1,2,3,10`，约 1.5 万字符）作为 system prompt，让 LLM 输出
   `{"positive","negative","params"}`，插件再按 `params` 的 steps / CFG / 尺寸补默认值。

找不到 skill 文件时按用户原文出图，不会报错；`/enhance` 会提示去哪里获取。

## Layout Enhancer

适合海报、封面、带标题或需要精确摆放元素的画面。流程严格按官方 skill：

1. LLM 只负责写 **draft JSON**：`width/height`、`bbox_coord_space="abs"`、
   `bbox_format="xyxy"`、`summary`、扁平的 `items`（`object`/`character`/`text`）
2. 坐标换算、越界校验、中心化 `yxyx` 压缩全部交给内置的官方
   `compile_prompt.py`，LLM 不碰坐标数学
3. 校验失败会把具体错误回传给 LLM 重写（默认重试 2 次），仍失败则退化为普通描述

画布尺寸以命令/配置为准，`compile_prompt` 会拒绝超出画布的框。

## 结构

```
astrbot_plugin_moss_image/
├── main.py             # 命令、两个 LLM 工具、结果组装
├── moss_client.py      # 与本机服务的 HTTP 客户端（生成 / 改图）
├── prompt_skill.py     # 提示词 skill 加载（不内置规则）
├── layout_enhancer.py  # Layout Enhancer（LLM draft + 校验重试）
├── prompt_parser.py    # --flag 解析、尺寸换算
├── compile_prompt.py   # 官方 Layout 编译器（原样内置）
└── _conf_schema.json   # 插件配置面板
```

## 常见问题

**「无法连接 MOSS-Image 服务」**
服务没起来。运行服务端目录下的 `run.ps1`（或装好的开机自启）；`/drawstatus` 可看连接状态。

**第一次出图 / 改图特别慢**
服务是懒加载的：第一个请求先加载模型（约 20 秒）。改图模型冷启动约 21 秒，
热态 1024²/12 步约 13 秒；空闲 10 分钟自动卸载并释放显存。

**改图报「改图功能不可用」**
服务端没配 `edit_adapter`。`/drawstatus` 会显示原因，
或 `GET /v1/edit/status` 看 `reason`。

**`/edit` 说没找到图片**
图片要和指令一起发，或引用（回复）一张带图的消息；部分适配器的引用只带 id，
插件会尝试 `get_msg` 取回，仍失败就请直接把图片和指令放在同一条消息里。

**Layout 总是失败**
换个更明确的描述，或把 `layout.max_retries` 调大。Layout 对文字框尺寸敏感，
标题太长时更适合放大画布（`--size 1536x1024`）。
