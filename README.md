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

**GitHub 开源搜索集成 (v2.1 新增)**
- Director 和 Architect 可主动搜索 GitHub 发现已有的开源解决方案
- 自动评估仓库质量（stars、活跃度、license、issue 关闭率）
- Builder 在实现阶段可搜索特定 npm/pip 包
- 支持仓库详情检查（README、文件结构、依赖声明）
- 可选 `GITHUB_TOKEN` 环境变量提升 API 速率限制

**搜索树与回溯机制 (v2.1 新增)**
- 架构设计阶段生成多个候选方案（而非线性单一路径）
- 基于 SWE-Search (ICLR 2025) 思路：分支-评估-选择-回溯
- 多样性过滤：自动去除本质相同的候选方案
- 失败时自动回溯到次优分支，而非从头重来
- 配置：`FORGE_SEARCH_TREE=true`，`FORGE_SEARCH_CANDIDATES=3`

**中途检查点 (v2.1 新增)**
- Builder 每 8 轮插入轻量方向检查（Process Reward Model 风格）
- 检测方向偏离、空转、过度工程等问题
- 低分时注入课程修正反馈，极低分时回滚到上一个好的检查点
- 配置：`FORGE_CHECKPOINTS=true`，`FORGE_CHECKPOINT_INTERVAL=8`

**动态 Constitution (v2.1 新增)**
- SPEC 阶段后，Director 根据项目特点生成针对性的 Agent 指令
- 例如 WebSocket 项目会给 Builder 注入异步错误处理指南
- 支持从失败中学习：自动生成防止同类问题的规则（Meta-Learning）
- 规则持久化到 `.autoforge/knowledge_base.json`，跨项目复用

**进化引擎 (v2.2 新增)**
- 跨项目工作流自我进化：每次运行是一个"基因组"，包含所有策略参数
- **策略记忆**：成功的工作流配置持久化到 `~/.autoforge/evolution_memory.json`
- **变异机制**：新项目继承最佳策略并施加小变异，探索更优配置
- **适应度追踪**：质量×完成率×测试通过率×成本效率 = 综合适应度
- **LLM 自省**：项目完成后 AI 分析什么有效什么可改进，指导下一代进化
- **交叉繁殖**：不同项目类型的成功策略可以交叉组合
- **MAP-Elites 式多样性**：按技术栈分 niche 保留多种优秀策略
- 配置：`FORGE_EVOLUTION=true`（默认开启）

**DSPy/OPRO 提示自优化 (v2.3 新增)**
- 灵感来自 Stanford DSPy 和 DeepMind OPRO：用 LLM 优化 LLM 的提示词
- **Thompson Sampling 选择**：在多个提示变体中自动平衡探索与利用
- **OPRO 优化**：积累足够数据后，LLM 分析当前提示词的表现并提出改进
- **AMPO 变异**：支持 focus_shift / simplify / elaborate / restructure 四种变异策略
- **适应度追踪**：每个变体记录使用次数、平均/最优/最差适应度
- 状态持久化到 `~/.autoforge/prompt_optimization/state.json`，跨项目累积优化
- 配置：`prompt_optimization_enabled=True`（默认开启）

**CodePRM 过程奖励模型 (v2.3 新增)**
- 灵感来自 ACL 2025 CodePRM：对代码生成的每一步打分，而非只看最终结果
- **步骤级评估**：PLANNING / FILE_CREATE / FILE_MODIFY / TEST_WRITE 等 8 种步骤类型
- **双信号融合**：LLM 判断（60%）+ 执行反馈（40%）= 综合过程奖励
- **执行验证**：自动运行 py_compile / node --check 验证语法，执行测试获取真实反馈
- **轨迹分析**：加权评估整条轨迹（后期步骤权重更高），自动识别瓶颈步骤
- **课程修正预警**：连续 3 步低分或趋势恶化时触发预警，提前介入
- 轨迹数据保存到 `.autoforge/prm_trajectories/`，供进化引擎分析
- 配置：`process_reward_enabled=True`（默认开启）

**RethinkMCTS 增强搜索 (v2.3 新增)**
- 灵感来自 RethinkMCTS (2024) 和 RPM-MCTS (2025)：用执行反馈矫正错误推理
- **完整 MCTS 循环**：SELECT（UCB1）→ EXPAND → SIMULATE → BACKPROPAGATE → REFINE
- **思维链修正（核心创新）**：执行失败时，LLM 分析思维链找到出错步骤，创建修正兄弟节点
- **过程奖励集成**：SIMULATE 阶段融合 CodePRM 信号（LLM 评估 60% + PRM 40%）
- **鲁棒选择**：最终选择访问次数最多的节点（而非最高值），更稳定
- 与原有 SearchTree 共存：架构探索用 SearchTree，具体实现用 MCTSSearchTree
- 配置：`mcts_enabled=True`，`mcts_max_iterations=9`

**三模块闭环优化**
- DSPy 优化提示 → RethinkMCTS 探索方案 → CodePRM 评估每步 → 适应度反馈回 DSPy
- 形成自动化的 "优化-探索-评估" 循环，每个项目都让系统变得更聪明

**EvoMAC 文本反向传播 (v2.4 新增)**
- 灵感来自 EvoMAC (ICLR 2025)：多 Agent 之间传递自然语言"梯度"
- **前向传播**：Builder 写代码 → Tester/Reviewer 测试/审查
- **反向传播**：LLM 从测试/审查结果中提取"文本梯度"，描述每个 Agent 应该如何改进
- **拓扑进化**：追踪哪些反馈路径有效，自动增强或剪枝 Agent 之间的通信边
- 配置：`evomac_enabled=True`

**SICA 自我改进编码智能体 (v2.4 新增)**
- 灵感来自 SICA (ICLR 2025 Workshop) + STO (NeurIPS 2025)：Agent 自己编辑自己的提示规则
- **性能分析**：每次运行后 LLM 分析质量数据，提出 constitution 文件的具体编辑建议
- **安全护栏**：保护关键文件不被修改（orchestrator.py, config.py 等），编辑必须通过验证
- **自动回滚**：如果改进后性能下降超过 10%，自动恢复到之前的版本
- 配置：`sica_enabled=True`

**跨项目 RAG 代码检索 (v2.4 新增)**
- 灵感来自 arXiv 2510.04905：从历史项目中检索相关代码片段，辅助新项目生成
- **BM25 + TF-IDF 混合检索**：无需外部向量模型，纯 Python 实现
- **代码感知分词**：识别 camelCase 和 snake_case，按函数/类粒度提取
- **质量加权**：高质量项目的代码片段在检索中获得更高权重
- 配置：`rag_enabled=True`

**形式化验证 (v2.4 新增)**
- 灵感来自 Vericoding：多层次代码验证（静态分析 → 类型检查 → 安全扫描 → LLM 形式化分析）
- **自动检测工具**：自动发现可用的 flake8、mypy、bandit、eslint 等工具
- **LLM 形式化分析**：检查不变量违反、资源泄漏、并发问题、逻辑错误
- 配置：`formal_verify_enabled=True`

**条件多 Agent 辩论 (v2.4 新增)**
- 灵感来自条件辩论 (ICLR 2025)：只在候选方案分数接近时触发辩论，避免不必要的开销
- **不确定性检测**：LLM 评估是否需要辩论（分数差 < 0.15 时触发）
- **奖励引导**：每轮辩论后 LLM 给论证打分，引导辩论方向
- **收敛停止**：单一立场胜出、奖励分差明显、或达到共识时自动终止
- 配置：`debate_enabled=True`

**RedCode 安全扫描 (v2.4 新增)**
- 灵感来自 RedCode (NeurIPS 2024)：生成代码的安全漏洞扫描
- **模式匹配**：15+ Python 规则、10+ JavaScript 规则，覆盖 OWASP Top 10 和 CWE 常见漏洞
- **依赖扫描**：集成 pip-audit 和 npm audit 检查已知漏洞
- **LLM 深度分析**：对关键文件进行逻辑级安全审查
- 配置：`security_scan_enabled=True`

**Reflexion 情景记忆 (v2.5 新增)**
- 灵感来自 Reflexion (NeurIPS 2023, Shinn et al.)：HumanEval +11%
- **语言强化学习**：Agent 失败后生成自然语言"反思"，而非简单重试
- **情景记忆**：跨任务、跨项目积累的失败反思库，避免重蹈覆辙
- **解析标签**：自动从错误信息提取失败类型（import_error、type_error 等）
- 配置：`reflexion_enabled=True`

**自适应推理计算 (v2.5 新增)**
- 灵感来自 "Scaling LLM Test-Time Compute" (ICLR 2025)：自适应分配比均匀 best-of-N 高效 4×
- **难度估计**：关键词信号 + 项目规模 + LLM 评估，混合判断任务复杂度
- **四级计算配置**：TRIVIAL → STANDARD → COMPLEX → EXTREME，自动调整重试次数、MCTS 深度、模型选择
- **自校准**：追踪预测 vs 实际难度，用指数移动平均自动修正偏差
- 配置：`adaptive_compute_enabled=True`

**LDB 块级调试器 (v2.5 新增)**
- 灵感来自 LDB (ACL 2024, Zhong et al.)：HumanEval +9.8%，GPT-4o 达到 98.2%
- **控制流分解**：将代码拆分为基本块（条件、循环、赋值、返回）
- **运行时追踪**：通过沙盒或 LLM 模拟追踪每个块的变量值
- **精确定位**：LLM 逐块验证，找到第一个输出不符预期的块
- **靶向修复**：只修改出错的块，而非盲目重写整个函数
- 配置：`ldb_debugger_enabled=True`

**推测执行流水线 (v2.5 新增)**
- 灵感来自 Speculative Actions (arXiv 2510.04371) 和 Sherlock (arXiv 2511.00330)
- **阶段重叠**：SPEC 完成时已提前创建好项目骨架、BUILD 完成时已准备好测试框架
- **投机验证**：推测性工作完成后验证是否与实际流水线兼容，冲突则丢弃
- **预计加速 20-40%**：减少阶段间的等待时间
- 配置：`speculative_enabled=True`

**层级任务分解 (v2.5 新增)**
- 灵感来自 Parsel (NeurIPS 2023)：竞赛题通过率比直接生成高 75%
- 灵感来自 CodePlan (ACM 2024)：仓库级依赖感知代码规划
- **函数级分解**：LLM 将复杂任务拆分为 3-15 个有依赖关系的函数规格
- **拓扑排序**：Kahn 算法确定实现顺序，叶子函数先实现
- **自底向上实现**：每个函数实现时已有依赖函数作为上下文
- 配置：`hierarchical_decomp_enabled=True`

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

# GitHub 集成（可选，提升搜索能力）
GITHUB_TOKEN=                    # GitHub Personal Access Token（提升 API 速率限制）

# 搜索树（v2.1 新增）
FORGE_SEARCH_TREE=true           # 启用多方案搜索树（架构阶段）
FORGE_SEARCH_CANDIDATES=3        # 每次分支生成的候选方案数

# 中途检查点（v2.1 新增）
FORGE_CHECKPOINTS=true           # 启用 Builder 中途方向检查
FORGE_CHECKPOINT_INTERVAL=8      # 每 N 轮检查一次方向

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
│   │   ├── agent_base.py    ← Agent 基类：tool-use 循环 + 检查点 + 动态 prompt
│   │   ├── search_tree.py   ← 搜索树：分支/评估/回溯（SWE-Search 风格）
│   │   ├── checkpoints.py   ← 中途检查点：Process Reward Model 式方向检查
│   │   ├── dynamic_constitution.py ← 动态 Constitution + Meta-Learning 知识库
│   │   ├── evolution.py     ← 进化引擎：跨项目工作流自我进化
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
│   │   ├── tools/           ← Agent 工具
│   │   │   ├── web.py       ← Web 搜索 + URL 抓取
│   │   │   ├── search.py    ← 代码搜索 (grep)
│   │   │   └── github_search.py ← GitHub 仓库/代码搜索
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
