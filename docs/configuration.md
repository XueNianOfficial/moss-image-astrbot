# 配置说明

## 服务端 `config.toml`

安装脚本会从 `config.example.toml` 生成它。所有键都可省略（有默认值），优先级：
**命令行参数 > 环境变量 `MOSS_<KEY>` > config.toml > 内置默认**。

```bash
python -m moss_server --config config.toml --port 8000
```

### 路径与模型

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `aitk_root` | `""` | ai-toolkit 目录，必须有 `toolkit/` 与 `extensions_built_in/` |
| `model_dir` | `""` | MOSS-Image diffusers 目录（含 `model_index.json`） |
| `dtype` | `bf16` | 基础权重精度 |
| `vae_path` | `""` | 单独指定 VAE；留空用 `<model_dir>/vae` |

### 量化与显存

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `quantize` | `true` | DiT 是否 int8 量化 |
| `qtype` | `convrot8` | ai-toolkit 的量化类型 |
| `quantize_te` / `qtype_te` | `true` / `convrot8` | 文本编码器同上 |
| `low_vram` | `true` | 组件在不推理时留在内存 |
| `layer_offloading` | `true` | 分层卸载总开关 |
| `layer_offloading_transformer_percent` | `0.0` | 0=用过的层留在显存；1=每步都从内存流式加载（最省显存最慢） |
| `layer_offloading_text_encoder_percent` | `1.0` | 文本编码器卸载比例，编码完就没用了，建议保持 1.0 |

显存紧张时的调整顺序：`layer_offloading_transformer_percent` 0.3 → 0.6 → 1.0。

### 生命周期

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `lazy_load` | `true` | 首个请求才加载模型 |
| `idle_unload_seconds` | `600` | 空闲多久卸载；`0` 表示常驻 |
| `warmup` | `true` | 加载后先跑一张 256² 预热（首次请求更快） |
| `generation_timeout` | `900` | 单次生成超时（秒） |
| `queue_timeout` | `900` | 排队超时（秒） |

### 生成默认值

`default_steps`、`default_guidance_scale`、`default_width`、`default_height`、
`default_max_sequence_length`、`default_uncond_skip_layers`（默认 `[22]`）、
`default_shift`（默认 `5.0`）、`max_pixels`。

请求里给了 `shift` / `uncond_skip_layers` 就只改当前 holder 的属性，不会重新加载模型。

### LoRA

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `loras` | `[]` | 启动时注册的文件路径；AstrBot 插件会自己注册，通常留空 |
| `lora_scale` | `0.7` | 注册时的默认强度 |
| `lora_mode` | `hook` | `hook`=forward hook 动态挂载；`merge`=写入权重（不推荐） |
| `lora_dirs` | `[]` | `GET /v1/loras` 的扫描目录 |

同一个文件只会有一条注册（按路径去重）；请求里出现未注册的名字会返回 **400**。

### 图像参考编辑（Moss-Image-Edit）

改图与文生图共用**同一个已加载的模型**：ai-toolkit 的 MOSS 模型本身就有参考图分支
（参考 token 走 t=0 调制、只与自身互相可见、K/V 跨步缓存）。改图请求只是翻转 holder 上的
`is_edit` / `kv_cache` / 视觉提示词编码三个开关、临时挂上编辑 LoRA，结束后翻回文生图状态，
**不重载任何权重**。适配器第一次使用时会融合成 ai-toolkit 的 `proj.qkv` 布局并缓存到
`output_dir/edit-lora/`。`edit_adapter` 留空即关闭改图。

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `edit_enabled` | `true` | 总开关；`--no-edit` 亦可 |
| `edit_adapter` | `""` | 编辑 LoRA 目录：`adapter_config.json` + `adapter_model.safetensors`（`hf download chinoll/Moss-Image-Edit --local-dir <dir>`） |
| `edit_steps` / `edit_guidance_scale` | `20` / `4.0` | 默认步数与 CFG（官方示例常用 12-50 / 4.0） |
| `edit_shift` | `5.0` | 采样 shift |
| `edit_uncond_skip_layers` | `[]` | CFG 跳过层；编辑 LoRA 训练时未跳过 |
| `edit_max_ref_images` | `4` | 参考图上限（1-8） |
| `edit_max_edge` | `1024` | 参考图缩放后的长边 |
| `edit_timeout` | `900` | 单次改图（含模型冷启动）的超时 |

CLI：`--edit-adapter`、`--no-edit`、`--edit-uncond-skip-layers`。
（`--edit-idle-unload` / `--no-edit-free-t2i` 已停用：只有一个模型、一个空闲计时器。）

改图与文生图是同一个模型，热态显存约 7 GiB（改图请求期间多占约 2-3 GiB 用于参考图
latent 与 K/V，请求结束即释放）。显存紧张时把 `edit_max_edge` 降到 768。

### HTTP

`host`、`port`、`api_key`（非空则要求 `Authorization: Bearer <key>`）、
`public_base_url`（返回 URL 模式时的前缀）、`output_dir`、`log_level`。

---

## 插件配置

在 AstrBot WebUI → 插件管理 → MOSS-Image 绘图 → 配置。保存后无需重启，重载插件即可。

| 段 | 键 | 说明 |
| --- | --- | --- |
| `server` | `base_url` / `api_key` / `timeout` | 服务地址与鉴权 |
| `generation` | `width` `height` `steps` `guidance` | 默认尺寸与采样 |
| | `negative_prompt` | **默认负面提示词**，每次出图都带 |
| | `shift` | `0`=跟随服务端；调低更贴近训练分布 |
| | `uncond_skip_layers` | 默认 `22`；`off`=不跳过（速度减半） |
| `enhance` | `enabled` `auto_enhance_below` `max_prompt_chars` | 是否自动改写短描述（默认 `false`） |
| `skill` | `enabled` `path` `sections` `max_chars` `guidance` `timeout` | 提示词 skill（见下） |
| `edit` | `enabled` `max_images` | 改图开关与参考图上限 |
| `layout` | `enabled` `max_retries` | 布局增强 |
| `lora` | `enabled` `allow_bot_choice` `max_weight` | LoRA 总开关与 AI 决策权限 |
| | `catalog_lines` | LoRA 目录文本（见下） |
| `behaviour` | `admin_only` `cooldown` `send_progress` `send_parameters` | 使用行为 |

### 提示词 skill

插件**不内置**任何提示词规则。规则来自
[moss-image-prompt-skill](https://github.com/Moeblack/moss-image-prompt-skill)（CC BY-NC-ND 4.0），
运行时从磁盘读取：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `skill.enabled` | `true` | 关掉后既不注入提示，也不做改写 |
| `skill.path` | `""` | 留空则找 `<AstrBot数据目录>/skills/moss-image-prompt-skill/SKILL.md`；也可直接填文件或目录 |
| `skill.sections` | `0,1,2,3,10` | 参与改写的章节；留空 = 全文（约 6.6 万字符，较贵） |
| `skill.max_chars` | `0` | 章节文本上限，`0` = 不截断（默认章节约 1.5 万字符） |
| `skill.guidance` | `true` | 在系统提示里要求模型出图前先读 SKILL.md |
| `skill.timeout` | `150` | 改写超时；超时回退原文，不会导致出图失败 |

两条使用路径：

1. **对话模型自己读**（默认）：`guidance` 注入「先读该文件、按章节取用」的指令，模型用文件工具
   自己读 skill，再带 prompt 调 `moss_draw_image`；
2. **插件代读**（`--enhance` 或 `enhance.enabled = true`）：插件取 `sections` 指定章节作为
   system prompt，让 LLM 按 skill 输出 `{"positive","negative","params"}`。

### LoRA 目录格式

```
名称 | 文件路径 | on/off | 权重 | 描述 | 触发词
```

```
# 注释行忽略
kemono | G:/loras/kemono.safetensors | on | 0.7 | 兽人画风：柔和赛璐璐上色、蓬松毛发。 |
mychar | G:/loras/mychar.safetensors | off | 1.0 | 角色「某角色」：白毛红眼、冰蓝渐变耳尾。 | 角色触发词
```

- 多条可以分行，也可以在同一行用 `;` 分隔
- `off` 的条目仍会以权重 0 注册，随时可被请求唤醒（hook 在 0 时直接短路，零开销）
- 描述会被注入系统提示，AI 据此判断用不用、用多重

---

## 开机自启

### Windows

```powershell
.\install-autostart.ps1           # 优先计划任务，被拒绝则回退到启动文件夹
.\install-autostart.ps1 -Remove   # 取消
```

### Linux (systemd)

```ini
# ~/.config/systemd/user/moss-image.service
[Unit]
Description=MOSS-Image API server
After=network-online.target

[Service]
WorkingDirectory=%h/moss-image-astrbot
ExecStart=%h/moss-image-astrbot/.ai-toolkit/.venv/bin/python -m moss_server --config %h/moss-image-astrbot/config.toml
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now moss-image
```

---

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `AI_TOOLKIT_DIR` | `run.ps1` / `run.bat` 找 ai-toolkit venv 的位置（`install.ps1` 也会写进 `.install.json`） |
| `MOSS_AI_MODEL` 等 `MOSS_<KEY>` | 覆盖任意服务端配置项，优先级高于 config.toml |
| `MOSS_TEST_LORA` | `tests/test_lora.py` 用来测试真实 LoRA 文件 |
| `MOSS_TEST_SKILL` | `tests/test_prompt_skill.py` / 插件集成测试用来读真实 SKILL.md |
| `MOSS_TEST_EDIT` | 设为 `1` 时插件集成测试会跑一次真实改图 |
| `ASTRBOT_APP_DIR` / `ASTRBOT_PLUGINS_DIR` | 集成测试定位 AstrBot 与插件目录 |
| `MOSS_SERVER_URL` | 集成测试指向的服务地址 |
