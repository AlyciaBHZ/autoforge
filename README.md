# AutoForge v2.0

AI 多 Agent 协作开发平台 —— 从想法到产品，不需要编程基础。

> **生成新项目**：一句话描述，AutoForge 帮你写出完整项目
> **接入已有项目**：导入任何阶段的项目，AI 帮你继续开发
> **代码审阅**：不想改代码？让 AI 审查后给出专业报告
> **移动端支持**：可选生成 iOS / Android 应用
> **全平台**：Windows / macOS / Linux 均可使用

---

## 安装

只需要 Python 3.11+ 和至少一个 LLM API Key（支持 Anthropic / OpenAI / Google Gemini）。

**一行安装（推荐）：**
```bash
pip install git+https://github.com/AlyciaBHZ/autoforge.git
```

安装完成后，直接在终端输入 `autoforge` 即可使用。首次运行会自动引导你选择 LLM 提供商、填入 API Key、选择模型和预算。

**也支持 pipx（隔离安装，不影响全局 Python 环境）：**
```bash
pipx install git+https://github.com/AlyciaBHZ/autoforge.git
```

**开发者模式（修改源码用）：**
```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
pip install -e .
```

---

## 四大核心功能

### 1. 生成新项目

告诉 AutoForge 你想做什么，它会帮你从零开始写出一个完整的项目。

```bash
autoforge generate "做一个有用户登录的 Todo App"
autoforge generate "Build a REST API for a bookstore with JWT auth"
autoforge generate "做一个个人作品集网站，极简风格，支持暗色模式"
```

8 个 AI Agent 会自动完成：需求分析 → 架构设计 → 代码编写 → 代码审查 → 测试验证 → 重构优化 → 打包交付。

### 2. 导入已有项目

已经有一个项目了？不管完成了多少，都可以导入让 AI 继续开发。

```bash
# 只分析，不修改
autoforge import ./my-project

# 分析并添加新功能
autoforge import ./my-project --enhance "加上暗色模式和用户设置页面"
```

AutoForge 会先扫描你的项目，理解技术栈和架构，然后在此基础上继续开发。

### 3. 代码审阅

让 AI 帮你做专业级的代码审查，发现 bug、安全漏洞、代码异味。

```bash
autoforge review ./my-project
```

输出一份详细报告，包括：
- 总体质量评分（1-10 分）
- 安全问题（SQL 注入、XSS、硬编码密钥等）
- 代码质量问题（重复代码、错误处理、命名规范）
- 每个问题的具体文件和行号
- 改进建议

在 **Developer 模式** 下，AI 还会自动帮你修复发现的问题。

### 4. 移动端生成

可选生成 iOS 和 Android 应用，支持 React Native 和 Flutter。

```bash
# 同时生成 iOS + Android
autoforge generate "做一个健身打卡 App" --mobile both

# 只生成 Android
autoforge generate "Build a recipe app" --mobile android
```

---

## 两种工作模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Developer** (默认) | AI 可以读写代码、运行命令 | 生成、导入、修复项目 |
| **Research** | AI 只读不写，只输出分析报告 | 代码审阅、技术评估、学习代码 |

```bash
# Developer 模式（默认）
autoforge review ./my-project --mode developer   # 审阅 + 自动修复

# Research 模式
autoforge review ./my-project --mode research    # 只审阅，不动代码
```

---

## 多模型 LLM 支持

AutoForge 支持三大主流 AI 提供商，你可以用手头已有的任何 API Key：

| 提供商 | 支持的模型 | 适用场景 |
|--------|-----------|---------|
| **Anthropic** | Claude Opus 4、Sonnet 4.5、Haiku 4.5 | 默认推荐，代码生成质量最优 |
| **OpenAI** | GPT-4o、GPT-4o-mini、o3、o4-mini | 已有 OpenAI Key 的用户 |
| **Google** | Gemini 2.5 Pro、Gemini 2.5 Flash、Gemini 2.0 Flash | 高性价比，Flash 系列极其便宜 |

### 混合使用

不同任务可以选不同提供商的模型，最大化性价比：

```toml
# ~/.autoforge/config.toml
[api]
anthropic_key = "sk-ant-..."
openai_key = "sk-..."
google_key = "AI..."

[models]
strong = "claude-opus-4-6"       # 复杂任务用 Opus（最强）
fast = "gemini-2.5-flash"        # 常规任务用 Gemini Flash（最便宜）
```

AutoForge 会根据模型名字自动识别提供商，无需额外配置。

### 各模型成本对比

| 提供商 | 模型 | 输入 $/百万 token | 输出 $/百万 token |
|--------|------|:---------:|:---------:|
| Anthropic | Claude Opus 4 | $15.0 | $75.0 |
| Anthropic | Sonnet 4.5 | $3.0 | $15.0 |
| Anthropic | Haiku 4.5 | $1.0 | $5.0 |
| OpenAI | GPT-4o | $2.5 | $10.0 |
| OpenAI | GPT-4o-mini | $0.15 | $0.6 |
| OpenAI | o3 | $10.0 | $40.0 |
| OpenAI | o4-mini | $1.1 | $4.4 |
| Google | Gemini 2.5 Pro | $1.25 | $10.0 |
| Google | Gemini 2.5 Flash | $0.15 | $0.6 |
| Google | Gemini 2.0 Flash | $0.10 | $0.4 |

### 配置方式

```bash
# 方式 1：运行配置向导（推荐，交互式选择提供商和模型）
autoforge setup

# 方式 2：环境变量
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AI...
export FORGE_MODEL_STRONG=gpt-4o          # 复杂任务用 GPT-4o
export FORGE_MODEL_FAST=gemini-2.5-flash  # 常规任务用 Gemini Flash
```

---

## Git 版本控制

AutoForge 内置 Git 版本控制系统，为多 Agent 并行开发提供隔离与合并能力。

### 自动版本管理

项目生成时，AutoForge 会自动：
1. **初始化 Git 仓库** — 创建 `.gitignore`，首次提交
2. **分支隔离** — 每个 Builder Agent 在独立的 Git Worktree 中工作
3. **自动合并** — 任务完成并通过审查后，自动合并到 main 分支
4. **冲突检测** — 合并冲突时自动回滚，防止代码损坏

### 并行开发工作流

```
任务 A ──→ [Worktree A] Builder 写代码 → Commit → Reviewer 审查 → 合并到 main
任务 B ──→ [Worktree B] Builder 写代码 → Commit → Reviewer 审查 → 合并到 main
任务 C ──→ [Worktree C] Builder 写代码 → Commit → Reviewer 审查 → 合并到 main
                                                              ↓
                                                     最终项目（所有代码已合并）
```

每个 Builder Agent 工作在自己的分支上，互不干扰。审查通过后自动合并到 main，如果审查不通过则在同一分支上修改后重新提交。

### Git Sync 工具

项目附带 `scripts/git_sync.py` 脚本，用于多分支开发同步：

```bash
python scripts/git_sync.py status             # 查看分支与 main 的差异
python scripts/git_sync.py changelog          # 查看自上次同步后的变更
python scripts/git_sync.py merge-main         # 将 main 合并到当前分支
python scripts/git_sync.py sync               # 完整同步：fetch + merge + push
python scripts/git_sync.py cherry-pick <sha>  # 选择性合并指定 commit
```

---

## 能做什么项目？

| 项目类型 | 示例 | 默认技术栈 |
|----------|------|-----------|
| **Web 全栈应用** | Todo App、博客、电商 | Next.js + TypeScript + Tailwind |
| **后端 API** | REST API、GraphQL 服务 | Express + TypeScript 或 Flask |
| **CLI 命令行工具** | 文件处理器、数据转换 | Python + Click 或 Node.js |
| **静态网站** | 个人主页、落地页 | Next.js SSG + Tailwind |
| **Go 微服务** | 高性能 API、gRPC 服务 | Go + Gin/Echo |
| **Java 应用** | Spring Boot API | Java + Spring Boot |
| **iOS / Android 应用** | 健身打卡、食谱管理 | React Native 或 Flutter |
| **桌面端应用** | Markdown 编辑器 | Electron + React |
| **Python 包/库** | PyPI 可发布库 | Python + pyproject.toml |

### 使用示例

```bash
# Web 全栈应用
autoforge generate "做一个心理咨询预约平台，有用户端和咨询师端"

# 后端 API
autoforge generate "Build a REST API for managing a bookstore with JWT auth"

# CLI 工具
autoforge generate "写一个命令行工具，批量压缩图片并转成 WebP 格式"

# Go 微服务
autoforge generate "Build a URL shortener microservice in Go with Redis"

# 移动端应用
autoforge generate "做一个记账 App，支持图表统计" --mobile both

# 导入已有项目并增强
autoforge import ./my-old-project --enhance "重构前端，加上响应式设计"

# 审阅代码
autoforge review ./teammate-project --mode research
```

---

## 零基础入门指南

完全不懂编程？没关系。跟着这几步走就行。

### 第 1 步：安装 Python

去 [python.org](https://www.python.org/downloads/) 下载 **Python 3.11 或更高版本**。

- **Windows 用户**：安装时一定要勾选 "Add Python to PATH"
- **macOS 用户**：`brew install python@3.12`
- **Linux 用户**：`sudo apt install python3.12`

验证安装：
```bash
python3 --version    # 应显示 Python 3.11.x 或更高
```

### 第 2 步：获取 API Key

三家选一个即可（也可以同时配置多个）：

| 提供商 | 获取地址 | Key 格式 |
|--------|---------|---------|
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/) | `sk-ant-...` |
| **OpenAI** | [platform.openai.com](https://platform.openai.com/api-keys) | `sk-...` |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com/apikey) | `AI...` |

注册账号 → 创建 API Key → 复制保存 → 充值余额（建议 $10 起步，够做 2-3 个项目）

### 第 3 步：安装 AutoForge

```bash
pip install git+https://github.com/AlyciaBHZ/autoforge.git
```

### 第 4 步：启动

```bash
autoforge
```

AutoForge 会用交互式菜单引导你：
```
Welcome to AutoForge v2.0

? What would you like to do?
  > Generate a new project        # 生成新项目
    Review an existing project    # 审阅已有项目
    Import & improve a project    # 导入并改进项目
    Configure settings            # 修改设置

? Operating mode:
  > Developer (will modify code)  # 开发模式（可改代码）
    Research (analysis only)      # 研究模式（只分析）

? Describe your project:
  > 做一个带登录功能的博客系统

? Budget limit (USD): 10.0
? Generate mobile app? No
```

选完后坐等就行。一般 5-15 分钟后，项目就在 `workspace/` 文件夹里了。

---

## 平台支持

| 平台 | 安装方式 | 状态 |
|------|---------|:----:|
| **Linux** | `pip install` | 完全支持 |
| **macOS** | `pip install` | 完全支持 |
| **Windows** | `pip install` | 完全支持 |
| **WSL** | `pip install` | 完全支持 |

---

## 工作原理

AutoForge 有 **3 条流水线**，由 **8 个 AI Agent** 协作完成。

### 流水线一：生成新项目

```
你的描述（一句话）
    │
    ▼
 ┌──────────────────────────────────────────────┐
 │  SPEC  ─  Director 分析需求，输出项目规格      │
 │           选技术栈、拆模块、定 MVP 范围         │
 ├──────────────────────────────────────────────┤
 │  BUILD ─  Architect 设计架构，生成任务          │
 │           Builder x N 并行写代码               │
 │           Reviewer 审查每个任务的代码            │
 ├──────────────────────────────────────────────┤
 │  VERIFY─  Tester 安装依赖、编译、运行测试       │
 │           失败 → 自动生成修复任务，最多重试 3 次  │
 ├──────────────────────────────────────────────┤
 │  REFACTOR  Reviewer 评估质量                   │
 │            Gardener 优化代码、修安全问题         │
 ├──────────────────────────────────────────────┤
 │  DELIVER─  生成 README，整理结构，输出成本报告   │
 └──────────────────────────────────────────────┘
    │
    ▼
 workspace/<项目名>/  ← 你的项目，拿去用！
```

### 流水线二：审阅已有项目

```
你的项目路径
    │
    ▼
 ┌──────────────────────────────────────────────┐
 │  SCAN  ─  Scanner 分析项目结构和技术栈         │
 ├──────────────────────────────────────────────┤
 │  REVIEW─  Reviewer 全面审查代码质量和安全性     │
 ├──────────────────────────────────────────────┤
 │  REFACTOR（仅 Developer 模式）                 │
 │           Gardener 自动修复发现的问题           │
 ├──────────────────────────────────────────────┤
 │  REPORT─  输出详细审阅报告                     │
 └──────────────────────────────────────────────┘
```

### 流水线三：导入并改进

```
你的项目路径 + 增强描述
    │
    ▼
 ┌──────────────────────────────────────────────┐
 │  SCAN   ─  Scanner 分析现有代码               │
 │  REVIEW ─  Reviewer 评估现状                  │
 │  ENHANCE─  Director 合并新需求，Builders 开发   │
 │  VERIFY ─  Tester 验证                        │
 │  REFACTOR─ Gardener 优化                      │
 │  DELIVER ─ 打包交付                            │
 └──────────────────────────────────────────────┘
```

### 8 个 Agent 的分工

| Agent | 角色 | 使用模型 | 做什么 |
|-------|------|----------|--------|
| **Director** | 产品经理 | Opus | 理解需求，拆解模块，决定 MVP 范围 |
| **Architect** | 架构师 | Opus | 设计目录结构、数据模型、API，生成任务 DAG |
| **Builder** | 开发工程师 | Sonnet | 写代码！可以多个并行工作 |
| **Reviewer** | 代码审查员 | Sonnet | 审查代码质量、安全性，给出评分 |
| **Tester** | 测试工程师 | Sonnet | 安装依赖、编译构建、运行测试 |
| **Gardener** | 重构专家 | Sonnet | 根据 Reviewer 反馈优化代码 |
| **Scanner** | 项目分析师 | Opus | 分析已有项目，逆向生成项目规格 |
| **DirectorFix** | 修复调度员 | Opus | 测试失败时，分析问题并生成修复任务 |

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
- 按模型分别计费（支持 Anthropic / OpenAI / Google 所有模型）
- 超出预算自动停止，交付已完成的部分

**断点恢复**
- 每个阶段完成后自动保存状态
- 中断后 `autoforge resume` 从上次断点继续
- 不浪费已完成的工作

**Research 模式**
- 所有写操作被 agent_base 层自动拦截
- Agent 只能读取文件和分析代码，不会修改任何东西
- 适合代码审阅、技术评估、学习别人的代码

---

## 配置系统

AutoForge 支持三层配置，优先级从高到低：

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 最高 | CLI 参数 (`--budget 5.0`) | 当次运行的临时覆盖 |
| 中 | 项目 `.env` 文件 | 项目级配置 |
| 低 | `~/.autoforge/config.toml` | 全局默认（setup 向导写入） |

### 首次设置（推荐方式）

```bash
autoforge setup
```

交互式向导会引导你配置 API Key、模型偏好和默认预算，保存到 `~/.autoforge/config.toml`。

### 全局配置文件

`~/.autoforge/config.toml`（由 setup 向导自动创建）：

```toml
[api]
anthropic_key = "sk-ant-..."    # Anthropic（三家选填，至少一个）
openai_key = "sk-..."           # OpenAI
google_key = "AI..."            # Google Gemini

[models]
strong = "claude-opus-4-6"           # 复杂任务（可跨提供商混搭）
fast = "gemini-2.5-flash"           # 常规任务（自动识别提供商）

[defaults]
budget = 10.0        # 预算上限（美元）
max_agents = 3       # 并行 Agent 数
docker = false       # 是否启用 Docker 沙盒
mode = "developer"   # 默认工作模式
```

### 项目级配置

项目目录下的 `.env` 文件（覆盖全局配置）：

```bash
# LLM API Key（至少填一个）
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...

FORGE_MODEL_STRONG=claude-opus-4-6
FORGE_MODEL_FAST=claude-sonnet-4-5-20250929
FORGE_BUDGET_LIMIT=10.0
FORGE_MAX_AGENTS=3
FORGE_DOCKER_ENABLED=true
FORGE_LOG_LEVEL=INFO

# 守护进程模式（可选）
FORGE_TELEGRAM_TOKEN=            # Telegram Bot Token（从 @BotFather 获取）
FORGE_TELEGRAM_ALLOWED_USERS=    # 允许的用户（逗号分隔，空=全部允许）
FORGE_WEBHOOK_ENABLED=false      # 启用 REST API
FORGE_WEBHOOK_PORT=8420          # API 端口
FORGE_WEBHOOK_SECRET=            # API 认证密钥
```

---

## CLI 完整参数

### 子命令

| 命令 | 用法 | 说明 |
|------|------|------|
| *(无参数)* | `autoforge` | 启动交互式菜单 |
| `setup` | `autoforge setup` | 运行/重新运行配置向导 |
| `generate` | `autoforge generate "描述"` | 生成新项目 |
| `review` | `autoforge review ./path` | 审阅已有项目 |
| `import` | `autoforge import ./path` | 导入并改进项目 |
| `status` | `autoforge status` | 查看项目状态 |
| `resume` | `autoforge resume` | 恢复上次中断的任务 |

### 全局选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--budget` | 10.0 | API 花费上限（美元） |
| `--agents` | 3 | 并行 Builder 数量 |
| `--model` | claude-sonnet-4-5-20250929 | 常规任务模型 |
| `--mode` | developer | 工作模式：`developer` 或 `research` |
| `--mobile` | none | 移动端目标：`none` / `ios` / `android` / `both` |
| `--verbose` | — | 显示详细日志 |
| `--log-level` | INFO | 日志级别 |

### 向后兼容

旧版用法依然有效（需在源码目录中）：

```bash
python forge.py "做一个 Todo App"      # 等同于 autoforge generate
python forge.py --resume               # 等同于 autoforge resume
python forge.py --status               # 等同于 autoforge status
```

---

## 项目结构

```
autoforge/                   ← pip install 后可用的 Python 包
├── pyproject.toml           ← 包定义 + 依赖
├── forge.py                 ← 旧版入口（向后兼容）
│
├── autoforge/               ← 核心包
│   ├── __init__.py          ← 版本号 + DATA_DIR
│   │
│   ├── cli/                 ← 交互式 CLI
│   │   ├── app.py           ← 主入口：子命令分发
│   │   ├── interactive.py   ← InquirerPy 交互菜单
│   │   ├── setup_wizard.py  ← 首次配置向导
│   │   └── display.py       ← Rich 显示组件
│   │
│   ├── engine/              ← 核心引擎
│   │   ├── orchestrator.py  ← 总编排器：管理 3 条流水线
│   │   ├── config.py        ← 配置管理 + 预算追踪
│   │   ├── llm_router.py    ← LLM 路由：多提供商自动切换（Anthropic/OpenAI/Google）
│   │   ├── agent_base.py    ← Agent 基类：tool-use 循环 + 模式过滤
│   │   ├── task_dag.py      ← 任务 DAG：依赖分析 + 调度
│   │   ├── lock_manager.py  ← 跨平台原子锁
│   │   ├── git_manager.py   ← Git worktree 管理
│   │   ├── sandbox.py       ← 沙盒：安全执行生成的代码
│   │   ├── project_registry.py ← SQLite 多项目管理
│   │   ├── daemon.py        ← 24/7 守护进程控制器
│   │   ├── deploy_guide.py  ← Vercel 部署指南生成
│   │   ├── channels/        ← 输入渠道
│   │   │   ├── telegram_bot.py ← Telegram 机器人
│   │   │   └── webhook.py   ← REST API 接口
│   │   └── agents/          ← 8 个 Agent 实现
│   │       ├── director.py  ← 产品经理 + 修复调度
│   │       ├── architect.py ← 架构师
│   │       ├── builder.py   ← 开发工程师
│   │       ├── reviewer.py  ← 代码审查员
│   │       ├── tester.py    ← 测试工程师
│   │       ├── gardener.py  ← 重构专家
│   │       └── scanner.py   ← 项目分析师
│   │
│   └── data/                ← 内置数据（随包分发）
│       ├── constitution/    ← Agent 行为规则
│       │   ├── CONSTITUTION.md
│       │   ├── agents/      ← 角色定义
│       │   ├── workflows/   ← 工作流定义
│       │   └── quality_gates.md
│       └── templates/       ← 项目模板
│           ├── react-native-mobile/
│           └── flutter-mobile/
│
├── scripts/                 ← 工具脚本
│   └── git_sync.py          ← Git 多分支同步工具
├── tests/                   ← Smoke test
├── docker/                  ← Docker 沙盒配置
└── workspace/               ← 生成的项目（运行时创建）
```

---

## 成本估算

| 项目复杂度 | 示例 | 预估成本 |
|-----------|------|---------|
| 简单 | Todo App、个人主页 | $2-3 |
| 中等 | 博客系统、预约平台 | $4-6 |
| 复杂 | 电商 MVP、多角色平台 | $7-10 |
| 审阅 | 中型项目代码审查 | $1-3 |
| 导入+增强 | 已有项目+新功能 | $3-8 |

默认预算 $10 足够完成一个中等复杂度的 MVP。

---

## 环境变量配置

在 `.env` 文件中配置：

```bash
# LLM API Key（至少填一个，可同时填多个）
ANTHROPIC_API_KEY=sk-ant-...     # Anthropic（Claude 系列）
OPENAI_API_KEY=sk-...            # OpenAI（GPT / o 系列）
GOOGLE_API_KEY=AI...             # Google（Gemini 系列）

# 可选：模型选择（自动识别提供商，可跨提供商混搭）
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
autoforge queue "做一个博客网站"
autoforge queue "Build a REST API" --budget 5.0

# 3. 启动守护进程
autoforge daemon start

# 4. 查看项目状态
autoforge projects
autoforge daemon status

# 5. 查看部署指南
autoforge deploy <project_id>
```

### Telegram 机器人

1. 在 Telegram 中找到 [@BotFather](https://t.me/botfather)，发送 `/newbot` 创建机器人
2. 把获得的 Token 填入 `.env` 的 `FORGE_TELEGRAM_TOKEN`
3. 启动守护进程：`autoforge daemon start`
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
autoforge daemon install
# 按提示执行 systemctl 命令
```

**macOS (launchd)：**
```bash
autoforge daemon install
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
autoforge deploy <project_id>
```

---

## FAQ

**Q: 我完全不会编程，能用吗？**
A: 可以。AutoForge 专为不懂编程的人设计。你只需要用自然语言描述想做什么，剩下的交给 AI。

**Q: 支持中文吗？**
A: 完全支持。中文、英文、中英混合都可以。

**Q: 每次运行要花多少钱？**
A: 取决于项目复杂度，一般 $2-10。用 `--budget` 参数可以设置上限，不会超支。

**Q: 生成的项目能直接上线吗？**
A: AutoForge 生成的是 MVP（最小可行产品），可以直接运行和演示。正式上线前建议人工 review 一下。

**Q: 中断了怎么办？**
A: 运行 `autoforge resume`，会从上次断点继续，不浪费已完成的工作。

**Q: 能指定用什么技术栈吗？**
A: 可以。在描述里写清楚，比如 "用 Flask + Vue 做一个..."，Director 会尊重你的选择。

**Q: 只能用 Anthropic (Claude) 吗？**
A: 不是。AutoForge 支持 Anthropic、OpenAI、Google Gemini 三大提供商。运行 `autoforge setup` 选择你的提供商，或者直接设置环境变量。不同任务还可以用不同提供商的模型混搭。

**Q: 需要 Docker 吗？**
A: 不需要。没有 Docker 也能正常运行。Docker 只是提供更安全的沙盒隔离。

**Q: Research 模式和 Developer 模式有什么区别？**
A: Research 模式下 AI 只分析不修改，输出分析报告。Developer 模式下 AI 会实际修改代码。审阅别人的代码建议用 Research 模式。

**Q: 导入项目会破坏我的原始代码吗？**
A: 不会。AutoForge 会先复制一份到 workspace 目录，在副本上工作，你的原始项目不会被修改。

**Q: 移动端生成后怎么编译？**
A: AutoForge 会生成完整的项目代码和配置文件，但编译需要安装对应平台工具（Android Studio / Xcode）。生成的项目 README 会说明具体步骤。

**Q: 可以在 Windows 上用吗？**
A: 可以。`pip install` 安装后，`autoforge` 命令在所有平台通用。

**Q: 守护进程模式是什么？**
A: 守护进程模式让 AutoForge 在后台 24/7 运行。你可以通过 Telegram、API 或命令行随时提交开发任务，AutoForge 会自动排队处理。适合同时管理多个项目。

**Q: 生成的项目怎么部署上线？**
A: AutoForge 会自动生成 `DEPLOY_GUIDE.md` 部署指南，包含 Vercel 部署的完整步骤、环境变量配置和域名设置建议。运行 `autoforge deploy <id>` 查看。

**Q: 可以同时做多个项目吗？**
A: 可以。用守护进程模式，通过 `autoforge queue` 或 Telegram `/build` 提交多个任务，它们会按顺序自动构建。

---

## AI 编码工具兼容

Clone 本仓库后，以下 AI 编码工具会**自动读取项目配置**：

| AI 工具 | 配置文件 | 状态 |
|---------|---------|:----:|
| **Claude Code** | `CLAUDE.md` | 自动加载 |
| **OpenAI Codex CLI** | `AGENTS.md` | 自动加载 |
| **Cursor** | `.cursor/rules/autoforge.mdc` | 自动加载 |
| **GitHub Copilot** | `.github/copilot-instructions.md` | 自动加载 |
| **Windsurf (Codeium)** | `.windsurfrules` | 自动加载 |
| **Aider** | `.aider.conf.yml` → `AGENTS.md` | 自动加载 |

---

## 系统要求

- **Python 3.11+**（必须）— [下载](https://python.org)
- **LLM API Key**（至少一个）：
  - Anthropic — [获取](https://console.anthropic.com/)
  - OpenAI — [获取](https://platform.openai.com/api-keys)
  - Google Gemini — [获取](https://aistudio.google.com/apikey)
- **Git**（推荐）— 项目生成时使用 worktree 隔离并行开发 — [下载](https://git-scm.com)
- **Docker**（可选）— 用于沙盒隔离执行

---

## License

MIT
