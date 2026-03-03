# AutoForge v2.0

AI 多 Agent 协作开发平台 —— 从想法到产品，不需要编程基础。

> **生成新项目**：一句话描述，AutoForge 帮你写出完整项目
> **接入已有项目**：导入任何阶段的项目，AI 帮你继续开发
> **代码审阅**：不想改代码？让 AI 审查后给出专业报告
> **移动端支持**：可选生成 iOS / Android 应用
> **全平台**：Windows / macOS / Linux 均可使用

---

## 30 秒上手

**macOS / Linux：**
```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
./setup.sh
source .venv/bin/activate
python forge.py            # 启动交互式向导
```

**Windows：**
```cmd
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
setup.bat
.venv\Scripts\activate.bat
python forge.py            :: 启动交互式向导
```

首次运行会启动设置向导，引导你填入 API Key、选择模型和预算。之后就可以通过菜单选择你想做的事情了。

---

## 四大核心功能

### 1. 生成新项目

告诉 AutoForge 你想做什么，它会帮你从零开始写出一个完整的项目。

```bash
python forge.py generate "做一个有用户登录的 Todo App"
python forge.py generate "Build a REST API for a bookstore with JWT auth"
python forge.py generate "做一个个人作品集网站，极简风格，支持暗色模式"
```

8 个 AI Agent 会自动完成：需求分析 → 架构设计 → 代码编写 → 代码审查 → 测试验证 → 重构优化 → 打包交付。

### 2. 导入已有项目

已经有一个项目了？不管完成了多少，都可以导入让 AI 继续开发。

```bash
# 只分析，不修改
python forge.py import ./my-project

# 分析并添加新功能
python forge.py import ./my-project --enhance "加上暗色模式和用户设置页面"
```

AutoForge 会先扫描你的项目，理解技术栈和架构，然后在此基础上继续开发。

### 3. 代码审阅

让 AI 帮你做专业级的代码审查，发现 bug、安全漏洞、代码异味。

```bash
python forge.py review ./my-project
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
python forge.py generate "做一个健身打卡 App" --mobile both

# 只生成 Android
python forge.py generate "Build a recipe app" --mobile android
```

---

## 两种工作模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Developer** (默认) | AI 可以读写代码、运行命令 | 生成、导入、修复项目 |
| **Research** | AI 只读不写，只输出分析报告 | 代码审阅、技术评估、学习代码 |

```bash
# Developer 模式（默认）
python forge.py review ./my-project --mode developer   # 审阅 + 自动修复

# Research 模式
python forge.py review ./my-project --mode research    # 只审阅，不动代码
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
python forge.py generate "做一个心理咨询预约平台，有用户端和咨询师端"

# 后端 API
python forge.py generate "Build a REST API for managing a bookstore with JWT auth"

# CLI 工具
python forge.py generate "写一个命令行工具，批量压缩图片并转成 WebP 格式"

# Go 微服务
python forge.py generate "Build a URL shortener microservice in Go with Redis"

# 移动端应用
python forge.py generate "做一个记账 App，支持图表统计" --mobile both

# 导入已有项目并增强
python forge.py import ./my-old-project --enhance "重构前端，加上响应式设计"

# 审阅代码
python forge.py review ./teammate-project --mode research
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

### 第 2 步：安装 Git

去 [git-scm.com](https://git-scm.com/) 下载安装。一路默认即可。

### 第 3 步：获取 API Key

1. 打开 [console.anthropic.com](https://console.anthropic.com/)
2. 注册账号
3. 点击 "API Keys" → "Create Key"
4. 复制生成的 key（以 `sk-ant-` 开头）
5. 充值余额（建议 $10 起步，够做 2-3 个项目）

### 第 4 步：安装 AutoForge

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge

# macOS / Linux
./setup.sh

# Windows
setup.bat
```

### 第 5 步：启动

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate.bat

# 启动（首次会引导你完成设置）
python forge.py
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

| 平台 | 安装脚本 | 激活虚拟环境 | 状态 |
|------|---------|------------|:----:|
| **Linux** | `./setup.sh` | `source .venv/bin/activate` | 完全支持 |
| **macOS** | `./setup.sh` | `source .venv/bin/activate` | 完全支持 |
| **Windows** | `setup.bat` | `.venv\Scripts\activate.bat` | 完全支持 |
| **WSL** | `./setup.sh` | `source .venv/bin/activate` | 完全支持 |

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
- 按模型（Opus/Sonnet/Haiku）分别计费
- 超出预算自动停止，交付已完成的部分

**断点恢复**
- 每个阶段完成后自动保存状态
- 中断后 `python forge.py resume` 从上次断点继续
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
python forge.py setup
```

交互式向导会引导你配置 API Key、模型偏好和默认预算，保存到 `~/.autoforge/config.toml`。

### 全局配置文件

`~/.autoforge/config.toml`（由 setup 向导自动创建）：

```toml
[api]
anthropic_key = "sk-ant-..."

[models]
strong = "claude-opus-4-6"           # 复杂任务
fast = "claude-sonnet-4-5-20250929"  # 常规任务

[defaults]
budget = 10.0        # 预算上限（美元）
max_agents = 3       # 并行 Agent 数
docker = false       # 是否启用 Docker 沙盒
mode = "developer"   # 默认工作模式
```

### 项目级配置

项目目录下的 `.env` 文件（覆盖全局配置）：

```bash
ANTHROPIC_API_KEY=sk-ant-...
FORGE_MODEL_STRONG=claude-opus-4-6
FORGE_MODEL_FAST=claude-sonnet-4-5-20250929
FORGE_BUDGET_LIMIT=10.0
FORGE_MAX_AGENTS=3
FORGE_DOCKER_ENABLED=true
FORGE_LOG_LEVEL=INFO
```

---

## CLI 完整参数

### 子命令

| 命令 | 用法 | 说明 |
|------|------|------|
| *(无参数)* | `python forge.py` | 启动交互式菜单 |
| `setup` | `python forge.py setup` | 运行配置向导 |
| `generate` | `python forge.py generate "描述"` | 生成新项目 |
| `review` | `python forge.py review ./path` | 审阅已有项目 |
| `import` | `python forge.py import ./path` | 导入并改进项目 |
| `status` | `python forge.py status` | 查看项目状态 |
| `resume` | `python forge.py resume` | 恢复上次中断的任务 |

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

v1 的用法依然有效：

```bash
python forge.py "做一个 Todo App"      # 等同于 generate
python forge.py --resume                # 等同于 resume
python forge.py --status                # 等同于 status
```

---

## 项目结构

```
autoforge/
├── forge.py                 ← 入口（调用 cli/app.py）
├── setup.sh                 ← macOS/Linux 一键安装
├── setup.bat                ← Windows 一键安装
├── requirements.txt         ← Python 依赖
│
├── cli/                     ← 交互式 CLI
│   ├── app.py               ← 主入口：子命令分发
│   ├── interactive.py       ← InquirerPy 交互菜单
│   ├── setup_wizard.py      ← 首次配置向导
│   └── display.py           ← Rich 显示组件（进度条、报告）
│
├── engine/                  ← 核心引擎
│   ├── orchestrator.py      ← 总编排器：管理 3 条流水线
│   ├── config.py            ← 配置管理 + 预算追踪
│   ├── llm_router.py        ← LLM 路由：Opus/Sonnet/Haiku 自动选择
│   ├── agent_base.py        ← Agent 基类：tool-use 循环 + 模式过滤
│   ├── task_dag.py          ← 任务 DAG：依赖分析 + 调度
│   ├── lock_manager.py      ← 跨平台原子锁
│   ├── git_manager.py       ← Git worktree 管理
│   ├── sandbox.py           ← 沙盒：安全执行生成的代码
│   └── agents/              ← 8 个 Agent 实现
│       ├── director.py      ← 产品经理 + 修复调度
│       ├── architect.py     ← 架构师
│       ├── builder.py       ← 开发工程师
│       ├── reviewer.py      ← 代码审查员
│       ├── tester.py        ← 测试工程师
│       ├── gardener.py      ← 重构专家
│       └── scanner.py       ← 项目分析师（v2.0 新增）
│
├── constitution/            ← 宪法层：Agent 行为规则
│   ├── CONSTITUTION.md      ← 总宪法
│   ├── agents/              ← 每个 Agent 的角色定义
│   │   ├── director.md
│   │   ├── architect.md
│   │   ├── builder.md
│   │   ├── reviewer.md
│   │   ├── tester.md
│   │   ├── gardener.md
│   │   └── scanner.md       ← v2.0 新增
│   ├── workflows/           ← 工作流定义
│   │   ├── spec.md
│   │   ├── build.md
│   │   ├── verify.md
│   │   ├── refactor.md
│   │   ├── deliver.md
│   │   ├── review.md        ← v2.0 新增
│   │   └── import.md        ← v2.0 新增
│   └── quality_gates.md     ← 质量门禁标准
│
├── templates/               ← 项目模板
│   ├── react-native-mobile/ ← React Native 模板
│   └── flutter-mobile/      ← Flutter 模板
│
├── docker/                  ← Docker 沙盒配置
├── tests/                   ← Smoke test（28 项自动验证）
├── workspace/               ← 生成的项目
└── examples/                ← 示例需求
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
A: 运行 `python forge.py resume`，会从上次断点继续，不浪费已完成的工作。

**Q: 能指定用什么技术栈吗？**
A: 可以。在描述里写清楚，比如 "用 Flask + Vue 做一个..."，Director 会尊重你的选择。

**Q: 需要 Docker 吗？**
A: 不需要。没有 Docker 也能正常运行。Docker 只是提供更安全的沙盒隔离。

**Q: Research 模式和 Developer 模式有什么区别？**
A: Research 模式下 AI 只分析不修改，输出分析报告。Developer 模式下 AI 会实际修改代码。审阅别人的代码建议用 Research 模式。

**Q: 导入项目会破坏我的原始代码吗？**
A: 不会。AutoForge 会先复制一份到 workspace 目录，在副本上工作，你的原始项目不会被修改。

**Q: 移动端生成后怎么编译？**
A: AutoForge 会生成完整的项目代码和配置文件，但编译需要安装对应平台工具（Android Studio / Xcode）。生成的项目 README 会说明具体步骤。

**Q: 可以在 Windows 上用吗？**
A: 可以。用 `setup.bat` 安装，然后一样用 `python forge.py` 运行。Windows / macOS / Linux 全平台支持。

---

## 系统要求

- **Python 3.11+**（必须）— [下载](https://python.org)
- **Git**（必须）— [下载](https://git-scm.com)
- **Anthropic API Key**（必须）— [获取](https://console.anthropic.com/)
- **Docker**（可选）— 用于沙盒隔离执行

---

## License

MIT
