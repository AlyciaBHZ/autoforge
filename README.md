# AutoForge

从一句话到一个完整项目 —— AI 多 Agent 协作自动开发框架。

> 你只需要描述想做什么，AutoForge 用 6 个 AI Agent 帮你把项目写出来。
> 支持 Windows / macOS / Linux。不需要编程基础。

---

## 30 秒上手

**macOS / Linux：**
```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
./setup.sh                       # 一键安装
# 编辑 .env，填入 ANTHROPIC_API_KEY
source .venv/bin/activate
python forge.py "做一个有用户登录的 Todo App"
```

**Windows：**
```cmd
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
setup.bat                        :: 一键安装
:: 编辑 .env，填入 ANTHROPIC_API_KEY
.venv\Scripts\activate.bat
python forge.py "做一个有用户登录的 Todo App"
```

就这么简单。AutoForge 会自动完成需求分析、架构设计、代码编写、代码审查、测试验证，最终在 `workspace/` 目录下交付一个完整的项目。

---

## 能做什么项目？

AutoForge 不只能做网页，还能生成多种类型的项目：

| 项目类型 | 示例 | 默认技术栈 | 沙盒内可编译 |
|----------|------|-----------|:----------:|
| **Web 全栈应用** | Todo App、博客、电商 | Next.js + TypeScript + Tailwind | Yes |
| **后端 API** | REST API、GraphQL 服务 | Express + TypeScript 或 Flask | Yes |
| **CLI 命令行工具** | 文件处理器、数据转换 | Python + Click 或 Node.js | Yes |
| **静态网站** | 个人主页、落地页 | Next.js SSG + Tailwind | Yes |
| **Go 微服务** | 高性能 API、gRPC 服务 | Go + Gin/Echo | Yes |
| **Java 应用** | Spring Boot API | Java + Spring Boot | Yes |
| **移动端代码骨架** | React Native / Flutter | RN + TypeScript | 需自行编译 |
| **桌面端代码骨架** | Electron / Tauri | Electron + React | 需自行编译 |
| **Python 包/库** | PyPI 可发布库 | Python + pyproject.toml | Yes |

### 使用示例

```bash
# Web 全栈应用
python forge.py "做一个心理咨询预约平台，有用户端和咨询师端"

# 后端 API
python forge.py "Build a REST API for managing a bookstore with JWT auth"

# CLI 工具
python forge.py "写一个命令行工具，批量压缩图片并转成 WebP 格式"

# Go 微服务
python forge.py "Build a URL shortener microservice in Go with Redis"

# 移动端骨架
python forge.py "Create a React Native fitness tracker app with step counting"

# 桌面端骨架
python forge.py "做一个 Electron 桌面端 Markdown 编辑器，支持实时预览"

# 静态网站
python forge.py "做一个个人作品集网站，极简风格，支持暗色模式"
```

> **关于移动端和桌面端**：AutoForge 会生成完整的项目代码和配置文件，但最终编译需要你安装对应的平台工具（如 Android Studio / Xcode / Electron 工具链）。生成的项目 README 会说明如何搭建编译环境。

---

## 零基础入门指南

完全不懂编程？没关系。跟着这 5 步走：

### 第 1 步：安装 Python

去 [python.org](https://www.python.org/downloads/) 下载 **Python 3.11 或更高版本**。

- **Windows 用户**：安装时勾选 "Add Python to PATH"（很重要！）
- **macOS 用户**：建议用 Homebrew 安装 `brew install python@3.12`
- **Linux 用户**：`sudo apt install python3.12`（Ubuntu/Debian）

验证安装：
```bash
python3 --version    # 应显示 Python 3.11.x 或更高
```

### 第 2 步：安装 Git

去 [git-scm.com](https://git-scm.com/) 下载安装。

- **Windows 用户**：安装 Git for Windows，一路默认即可
- **macOS 用户**：`brew install git` 或者安装 Xcode Command Line Tools
- **Linux 用户**：`sudo apt install git`

### 第 3 步：获取 API Key

1. 打开 [console.anthropic.com](https://console.anthropic.com/)
2. 注册账号
3. 点击 "API Keys" → "Create Key"
4. 复制生成的 key（以 `sk-ant-` 开头）
5. 充值一点余额（建议 $10 起步，够做 2-3 个项目）

### 第 4 步：安装 AutoForge

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge

# macOS / Linux
./setup.sh

# Windows
setup.bat
```

然后用任意文本编辑器打开 `.env` 文件，把 `sk-ant-xxx` 替换成你的真实 API Key。

### 第 5 步：描述你想做的，然后坐等

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate.bat

# 然后运行
python forge.py "用中文或英文描述你想做的项目"
```

AutoForge 会显示实时进度。一般 5-15 分钟后，你的项目就在 `workspace/` 文件夹里了。

---

## 平台支持

| 平台 | 安装脚本 | 激活虚拟环境 | 状态 |
|------|---------|------------|:----:|
| **Linux** | `./setup.sh` | `source .venv/bin/activate` | 完全支持 |
| **macOS** | `./setup.sh` | `source .venv/bin/activate` | 完全支持 |
| **Windows** | `setup.bat` | `.venv\Scripts\activate.bat` | 完全支持 |
| **WSL** | `./setup.sh` | `source .venv/bin/activate` | 完全支持 |

---

## 工作原理

AutoForge 的核心是一个 **5 阶段流水线**，由 **6 个 AI Agent** 协作完成：

```
用户输入（自然语言）
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 1: SPEC（需求分析）                        │
│  Director Agent (Opus) 分析需求 → 输出 spec.json  │
│  • 理解你想做什么                                  │
│  • 选择技术栈（Next.js? Flask? Go? 等）           │
│  • 判断项目类型（Web? API? CLI? 移动端?）          │
│  • 拆分为独立模块                                  │
│  • 明确 MVP 范围，砍掉不必要的功能                  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 2: BUILD（并行开发）                       │
│  Architect Agent (Opus) 设计架构 → 生成任务 DAG    │
│  Builder Agents (Sonnet x N) 并行编写代码          │
│  Reviewer Agent (Sonnet) 审查每个任务的代码         │
│  • Architect 设计目录结构、数据模型、API 接口       │
│  • 任务按依赖关系排列成 DAG（有向无环图）           │
│  • 多个 Builder 同时认领不同任务，互不干扰          │
│  • 每个 Builder 在独立的 Git worktree 中工作        │
│  • Reviewer 审查通过后才合并代码                    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 3: VERIFY（测试验证）                      │
│  Tester Agent (Sonnet) 在沙盒中运行测试            │
│  • 安装依赖                                       │
│  • 编译/构建项目                                   │
│  • 运行测试用例                                    │
│  • 失败 → Director 生成修复任务 → Builder 修复      │
│  • 最多重试 3 次                                   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 4: REFACTOR（重构优化）                    │
│  Reviewer 评估代码质量                             │
│  Gardener Agent (Sonnet) 执行优化                  │
│  • 消除重复代码                                    │
│  • 修复安全问题                                    │
│  • 改进错误处理                                    │
│  • 重构后回归测试确保不引入新 bug                   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 5: DELIVER（打包交付）                     │
│  • 生成 README                                    │
│  • 整理项目结构                                    │
│  • 输出成本报告                                    │
└─────────────────────────────────────────────────┘
    │
    ▼
workspace/<项目名>/  ← 你的项目，拿去用！
```

### 6 个 Agent 的分工

| Agent | 角色 | 使用模型 | 做什么 |
|-------|------|----------|--------|
| **Director** | 产品经理 | Opus | 理解需求，拆解模块，决定 MVP 范围 |
| **Architect** | 架构师 | Opus | 设计目录结构、数据模型、API，生成任务 DAG |
| **Builder** | 开发工程师 | Sonnet | 写代码！可以多个并行工作 |
| **Reviewer** | 代码审查员 | Sonnet | 审查代码质量、安全性，给出评分 |
| **Tester** | 测试工程师 | Sonnet | 安装依赖、编译构建、运行测试 |
| **Gardener** | 重构专家 | Sonnet | 根据 Reviewer 反馈优化代码 |

### 关键技术特性

**任务 DAG 调度**
- 任务之间有依赖关系，形成有向无环图
- 没有依赖的任务可以并行执行
- 每个 Agent 同一时间只能认领 1 个任务（原子锁保证）

**Git Worktree 隔离**
- 每个 Builder 在独立的 Git 分支/worktree 中工作
- 完成后合并到 main 分支
- 避免多 Agent 同时编辑同一文件导致冲突

**沙盒执行**
- 生成的代码在沙盒（Docker 或子进程）中执行
- Docker 模式：无网络访问、内存/CPU 限制
- 子进程模式：不需要 Docker，直接在本地执行

**预算控制**
- 实时追踪每次 LLM 调用的 token 消耗
- 按模型（Opus/Sonnet）分别计费
- 超出预算自动停止，交付已完成的部分

**断点恢复**
- 每个阶段完成后自动保存状态
- 中断后 `--resume` 从上次断点继续
- 不浪费已完成的工作

**跨平台支持**
- macOS / Linux / Windows 全平台支持
- Windows 使用 `setup.bat`，macOS/Linux 使用 `setup.sh`
- 锁管理器自动适配（POSIX symlink / Windows 文件锁）

---

## 项目结构

```
autoforge/
├── forge.py                 ← 入口：接收你的需求，启动流水线
├── setup.sh                 ← macOS/Linux 一键安装
├── setup.bat                ← Windows 一键安装
├── .env.example             ← API Key 模板
├── requirements.txt         ← Python 依赖
│
├── engine/                  ← 核心引擎
│   ├── orchestrator.py      ← 总编排器：管理 5 阶段流水线
│   ├── config.py            ← 配置管理 + 预算追踪
│   ├── llm_router.py        ← LLM 路由：Opus/Sonnet 自动选择
│   ├── agent_base.py        ← Agent 基类：tool-use 循环
│   ├── task_dag.py          ← 任务 DAG：依赖分析 + 调度
│   ├── lock_manager.py      ← 跨平台原子锁
│   ├── git_manager.py       ← Git worktree 管理
│   ├── sandbox.py           ← 沙盒：安全执行生成的代码
│   ├── project_registry.py  ← SQLite 多项目管理
│   ├── daemon.py            ← 24/7 守护进程控制器
│   ├── deploy_guide.py      ← Vercel 部署指南生成
│   ├── channels/            ← 输入渠道
│   │   ├── telegram_bot.py  ← Telegram 机器人
│   │   └── webhook.py       ← REST API 接口
│   └── agents/              ← 6 个 Agent 实现
│       ├── director.py
│       ├── architect.py
│       ├── builder.py
│       ├── reviewer.py
│       ├── tester.py
│       └── gardener.py
│
├── constitution/            ← 宪法层：Agent 行为规则
├── services/                ← 系统服务配置（systemd / launchd）
├── docker/                  ← Docker 沙盒配置（Node.js + Python + Go + Java）
├── templates/               ← 项目模板（供 Agent 参考）
├── tests/                   ← Smoke test（31 项自动验证）
├── workspace/               ← 生成的项目（gitignore）
└── examples/                ← 示例需求
```

---

## CLI 完整参数

### 直接模式（一次性构建）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `description` | — | 项目需求描述（自然语言，中英文均可） |
| `--budget` | 10.0 | API 花费上限（美元） |
| `--agents` | 3 | 并行 Builder 数量 |
| `--model` | claude-sonnet-4-5-20250929 | 常规任务使用的模型 |
| `--resume` | — | 继续上次中断的任务 |
| `--status` | — | 查看当前所有项目状态 |
| `--verbose` | — | 显示详细日志 |
| `--log-level` | INFO | 日志级别（DEBUG/INFO/WARNING/ERROR） |

### 守护进程模式

| 命令 | 说明 |
|------|------|
| `python forge.py daemon start` | 启动 24/7 守护进程 |
| `python forge.py daemon stop` | 停止守护进程 |
| `python forge.py daemon status` | 查看守护进程状态 |
| `python forge.py daemon install` | 安装为系统服务 |
| `python forge.py queue "描述"` | 添加项目到构建队列 |
| `python forge.py queue "描述" --budget 5.0` | 指定预算添加项目 |
| `python forge.py projects` | 查看所有项目列表 |
| `python forge.py deploy <project_id>` | 查看项目部署指南 |

---

## 成本估算

| 项目复杂度 | 示例 | 预估成本 |
|-----------|------|---------|
| 简单 | Todo App、个人主页 | $2-3 |
| 中等 | 博客系统、预约平台 | $4-6 |
| 复杂 | 电商 MVP、多角色平台 | $7-10 |

默认预算 $10 足够完成一个中等复杂度的 MVP。

---

## 环境变量配置

在 `.env` 文件中配置：

```bash
# 必填
ANTHROPIC_API_KEY=sk-ant-...

# 可选：模型选择
FORGE_MODEL_STRONG=claude-opus-4-6           # 复杂任务（需求分析、架构设计）
FORGE_MODEL_FAST=claude-sonnet-4-5-20250929  # 常规任务（编码、审查、测试）

# 可选：预算和执行
FORGE_BUDGET_LIMIT=10.0          # 预算上限（美元）
FORGE_MAX_AGENTS=3               # 最大并行 Agent 数
FORGE_DOCKER_ENABLED=true        # 启用 Docker 沙盒（setup 脚本会自动设置）
FORGE_LOG_LEVEL=INFO             # 日志级别

# 守护进程模式（可选）
FORGE_TELEGRAM_TOKEN=            # Telegram Bot Token（从 @BotFather 获取）
FORGE_TELEGRAM_ALLOWED_USERS=    # 允许的用户（逗号分隔，空=全部允许）
FORGE_WEBHOOK_ENABLED=false      # 启用 REST API
FORGE_WEBHOOK_PORT=8420          # API 端口
FORGE_WEBHOOK_SECRET=            # API 认证密钥
```

---

## 24/7 守护进程模式

AutoForge 可以作为后台服务持续运行，通过 Telegram 或 API 接收开发请求，自动排队、构建、通知。

### 架构

```
用户 ──→ Telegram Bot ──→ ┌──────────┐     ┌─────────────┐
                          │  项目队列  │ ──→ │ AutoForge   │ ──→ workspace/
用户 ──→ Webhook API ──→  │ (SQLite)  │     │  Pipeline   │
                          └──────────┘     └─────────────┘
用户 ──→ CLI queue   ──→       ↑                   │
                              │                   ▼
                         完成/失败通知       DEPLOY_GUIDE.md
```

### 快速开始

```bash
# 1. 配置 .env（添加 Telegram Token 或启用 Webhook）

# 2. 通过 CLI 添加项目到队列
python forge.py queue "做一个博客网站"
python forge.py queue "Build a REST API" --budget 5.0

# 3. 启动守护进程
python forge.py daemon start

# 4. 查看项目状态
python forge.py projects
python forge.py daemon status

# 5. 查看部署指南
python forge.py deploy <project_id>
```

### Telegram 机器人

1. 在 Telegram 中找到 [@BotFather](https://t.me/botfather)，发送 `/newbot` 创建机器人
2. 把获得的 Token 填入 `.env` 的 `FORGE_TELEGRAM_TOKEN`
3. 启动守护进程：`python forge.py daemon start`
4. 在 Telegram 中和你的机器人对话：

| 命令 | 说明 |
|------|------|
| `/build 做一个博客` | 提交开发任务 |
| `/build $5 做一个博客` | 指定预算提交任务 |
| `/status` | 查看所有项目 |
| `/queue` | 查看当前队列 |
| `/budget` | 查看总花费 |
| `/cancel <id>` | 取消排队中的项目 |
| `/deploy <id>` | 获取部署指南 |

### Webhook REST API

启用后（`FORGE_WEBHOOK_ENABLED=true`），可以通过 HTTP API 管理项目：

```bash
# 提交新项目
curl -X POST http://localhost:8420/api/build \
  -H "Authorization: Bearer YOUR_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"description": "做一个博客网站", "budget": 5.0}'

# 查看所有项目
curl http://localhost:8420/api/projects \
  -H "Authorization: Bearer YOUR_SECRET"

# 查看项目详情
curl http://localhost:8420/api/projects/<id> \
  -H "Authorization: Bearer YOUR_SECRET"

# 获取部署指南
curl http://localhost:8420/api/projects/<id>/deploy \
  -H "Authorization: Bearer YOUR_SECRET"

# 取消排队中的项目
curl -X DELETE http://localhost:8420/api/projects/<id> \
  -H "Authorization: Bearer YOUR_SECRET"

# 健康检查
curl http://localhost:8420/api/health
```

### 安装为系统服务（开机自启）

**Linux (systemd)：**
```bash
python forge.py daemon install
# 按提示执行 systemctl 命令
```

**macOS (launchd)：**
```bash
python forge.py daemon install
# 按提示执行 launchctl 命令
```

**Windows：**
使用 Task Scheduler 或 NSSM 配置为服务。

---

## 部署到 Vercel

AutoForge 生成的 npm/serverless 前端项目会自动附带 `DEPLOY_GUIDE.md`，包含：

1. **推送到 GitHub** — 完整的 git 命令
2. **Vercel 部署** — 自动检测框架（Next.js/Vite/SvelteKit 等），一键部署
3. **环境变量配置** — 每个变量的获取方式和说明
4. **域名设置** — 性价比方案推荐：
   - 免费：`*.vercel.app` 子域名
   - 低成本：Cloudflare (~$9/年)、Porkbun (~$9/年)、Namecheap (~$9-13/年)
5. **Vercel 免费额度** — 100 GB 带宽/月，Serverless Functions，Preview Deployments

```bash
# 查看某个项目的部署指南
python forge.py deploy <project_id>
```

---

## 系统要求

- **Python 3.11+**（必须）— [下载地址](https://python.org)
- **Git**（必须）— [下载地址](https://git-scm.com)
- **Anthropic API Key**（必须）— [获取地址](https://console.anthropic.com/)
- **Docker**（可选）— 用于沙盒隔离执行，没有也能跑

---

## FAQ

**Q: 生成的项目能直接上线吗？**
A: AutoForge 生成的是 MVP（最小可行产品），可以直接运行和演示。正式上线前建议人工 review 一遍安全性和性能。

**Q: 支持中文描述吗？**
A: 完全支持。你用中文、英文、中英混合都可以。

**Q: 每次运行要花多少钱？**
A: 取决于项目复杂度，一般 $2-10。用 `--budget` 参数可以设置上限。

**Q: 中断了怎么办？**
A: 运行 `python forge.py --resume`，会从上次断点继续。

**Q: 能指定用什么技术栈吗？**
A: 可以。在描述里写清楚，比如 "用 Flask + Vue 做一个..."，Director 会尊重你的选择。

**Q: 需要 Docker 吗？**
A: 不需要。没有 Docker 也能正常运行（用子进程模式执行）。Docker 只是提供更安全的沙盒隔离。

**Q: 可以在 Windows 上用吗？**
A: 可以。用 `setup.bat` 安装，然后一样用 `python forge.py` 运行。

**Q: 生成的代码质量怎么样？**
A: 有 Reviewer Agent 做代码审查，有 Tester Agent 做自动测试，有 Gardener Agent 做重构优化。质量有保障。

**Q: 守护进程模式是什么？**
A: 守护进程模式让 AutoForge 在后台 24/7 运行。你可以通过 Telegram、API 或命令行随时提交开发任务，AutoForge 会自动排队处理。适合同时管理多个项目。

**Q: 生成的项目怎么部署上线？**
A: AutoForge 会自动生成 `DEPLOY_GUIDE.md` 部署指南，包含 Vercel 部署的完整步骤、环境变量配置和域名设置建议。运行 `python forge.py deploy <id>` 查看。

**Q: 可以同时做多个项目吗？**
A: 可以。用守护进程模式，通过 `python forge.py queue` 或 Telegram `/build` 提交多个任务，它们会按顺序自动构建。

---

## AI 编码工具兼容

Clone 本仓库后，以下 AI 编码工具会**自动读取项目配置**，无需额外设置即可理解项目并开始工作：

| AI 工具 | 配置文件 | 状态 |
|---------|---------|:----:|
| **Claude Code** | `CLAUDE.md` | 自动加载 |
| **OpenAI Codex CLI** | `AGENTS.md` | 自动加载 |
| **OpenCode** | `AGENTS.md` | 自动加载 |
| **Cursor** | `.cursor/rules/autoforge.mdc` | 自动加载 |
| **GitHub Copilot** | `.github/copilot-instructions.md` | 自动加载 |
| **Windsurf (Codeium)** | `.windsurfrules` | 自动加载 |
| **Aider** | `.aider.conf.yml` → `AGENTS.md` | 自动加载 |

**使用方式**：Clone 仓库 → 用你喜欢的 AI 编码工具打开 → AI 自动理解项目 → 开始工作。

---

## 宪法系统

AutoForge 通过 `constitution/` 目录中的规则文件控制 Agent 行为：

```
constitution/
├── CONSTITUTION.md          ← 总宪法：5 大原则 + 硬门禁规则
├── agents/                  ← 每个 Agent 的角色定义和行为规范
│   ├── director.md
│   ├── architect.md
│   ├── builder.md
│   ├── reviewer.md
│   ├── tester.md
│   └── gardener.md
├── workflows/               ← 5 个阶段的工作流定义
│   ├── spec.md
│   ├── build.md
│   ├── verify.md
│   ├── refactor.md
│   └── deliver.md
└── quality_gates.md         ← 阶段间的质量门禁标准
```

你可以修改这些文件来自定义 Agent 的行为，比如偏好特定技术栈。

---

## License

MIT
