```
     _         _        _____
    / \  _   _| |_ ___ |  ___|__  _ __ __ _  ___
   / _ \| | | | __/ _ \| |_ / _ \| '__/ _` |/ _ \
  / ___ \ |_| | || (_) |  _| (_) | | | (_| |  __/
 /_/   \_\__,_|\__\___/|_|  \___/|_|  \__, |\___|
                                       |___/
```

**一行命令，六个智能体，生成完整可运行的软件项目。**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-217%20checks-brightgreen.svg)](tests/)
[![Engines](https://img.shields.io/badge/engines-47%20modules-orange.svg)](autoforge/engine/)

[English](docs/README_EN.md) | [开发者文档](CLAUDE.md)

---

## 快速开始

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
pip install -e .                              # 安装（从源码）
forgeai                                       # 启动交互式会话
```

首次启动自动引导配置 API Key（Anthropic / OpenAI / Google 任选一家）、GitHub 环境和运行模式。之后直接进入会话，描述项目即开始构建。

<details>
<summary>可选依赖</summary>

```bash
pip install -e ".[openai]"       # OpenAI 支持
pip install -e ".[google]"       # Google Gemini 支持
pip install -e ".[search]"       # Web 搜索能力
pip install -e ".[channels]"     # Telegram / Webhook 频道
pip install -e ".[all]"          # 全部安装
```

</details>

---

## 它能做什么

AutoForge 通过 6 个专业化 AI 智能体协作，经过 5 阶段流水线，将一句自然语言描述转化为完整的代码项目。需求分析、架构设计、并行代码生成、自动化测试、代码审查、重构优化、部署打包，全程自动完成。

**核心特性：**
- 全栈 Web 应用、API 服务、CLI 工具、移动端应用，一行命令搞定
- 多模型多厂商支持，可跨厂商混搭（强模型做规划，快模型写代码）
- 预算控制与实时成本追踪（`--budget 5.00`）
- 沙盒执行（Docker 或子进程），安全运行生成的代码
- 断点恢复，中断后继续不丢失进度
- 24/7 守护进程模式，支持构建队列、Telegram 机器人和 REST API

---

## 架构

```
 "做一个带登录的 Todo App"
          |
          v
 ┌─────────────────────────────────────────────┐
 │  SPEC      Director 分析需求、拆解模块       │
 ├─────────────────────────────────────────────┤
 │  BUILD     Architect 设计架构                │
 │            Builder 并行写代码，Reviewer 审查  │
 ├─────────────────────────────────────────────┤
 │  VERIFY    Tester 安装依赖、构建、运行测试    │
 │            失败自动生成修复任务               │
 ├─────────────────────────────────────────────┤
 │  REFACTOR  Gardener 优化代码质量             │
 ├─────────────────────────────────────────────┤
 │  DELIVER   生成 README、整理结构、成本报告    │
 └─────────────────────────────────────────────┘
          |
          v
    workspace/my-todo-app/
```

| 智能体 | 角色 | 模型层级 |
|--------|------|----------|
| **Director** | 需求分析与范围界定 | Strong（强模型） |
| **Architect** | 系统设计与任务依赖图 | Strong（强模型） |
| **Builder** | 代码生成（可并行） | Fast（快模型） |
| **Reviewer** | 代码审查与评分 | Fast（快模型） |
| **Tester** | 构建、测试、自动修复循环 | Fast（快模型） |
| **Gardener** | 重构与安全修复 | Fast（快模型） |

---

## 支持的 LLM 提供商

| 提供商 | 环境变量 | 推荐模型 |
|--------|----------|----------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Opus 4（强）、Claude Sonnet 4（快） |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o / o3（强）、GPT-4o-mini（快） |
| **Google** | `GOOGLE_API_KEY` | Gemini 2.5 Pro（强）、Gemini 2.5 Flash（快） |

也可以通过 `~/.autoforge/config.toml` 统一管理密钥。支持跨厂商混搭：

```bash
export FORGE_MODEL_STRONG=gpt-4o          # 强模型用 OpenAI
export FORGE_MODEL_FAST=gemini-2.5-flash  # 快模型用 Google
```

---

## 使用方法

```bash
# 交互式会话（推荐）
forgeai                           # 选择模式 → 描述项目 → 开始构建

# 直接生成
forgeai generate "用 Flask + Vue 做一个书店管理系统，带 JWT 认证"
forgeai generate "SaaS 产品落地页" --budget 3.00

# 管理运行
forgeai status                    # 查看所有项目
forgeai resume                    # 恢复中断的任务

# 守护进程模式（24/7 后台服务）
forgeai daemon start
forgeai queue "支持 Markdown 的博客系统"
forgeai projects
forgeai deploy <project_id>
```

<details>
<summary>成本估算</summary>

| 复杂度 | 示例 | 预估成本 |
|--------|------|:--------:|
| 简单 | Todo App、落地页 | $2–3 |
| 中等 | 博客系统、预约平台 | $4–6 |
| 复杂 | 电商 MVP、多角色平台 | $7–10 |

默认预算上限 $10，可通过 `--budget` 覆盖。

</details>

---

## 智能引擎

AutoForge 内置多个智能引擎，在代码生成全流程中自动协作：

- **MCTS 搜索树** — 架构方案探索与择优，基于执行反馈动态修正思维链
- **自然语言梯度反馈** — Agent 间通过文本"反向传播"互相优化输出质量
- **过程奖励模型** — 逐步评估代码生成质量，而非仅看最终结果
- **自适应算力分配** — 根据任务难度动态调整推理深度和资源投入
- **语言强化学习** — 从失败中提取经验，下次重试时自动规避已知错误
- **块级故障定位** — 精确定位代码缺陷到代码块级别
- **函数级任务分解** — 将复杂需求拆解为可独立验证的函数级子任务
- **阶段预执行** — 流水线各阶段重叠并行，加速整体构建

---

## 学术与科研能力

AutoForge 内置完整的学术科研流水线，可作为 AI 驱动的科研助手使用。涵盖从论文阅读、形式化验证、自主发现到论文撰写的全流程。

### 端到端文章推理

> 输入任意一篇论文，自动完成：解析 → 理论图谱构建 → 声明验证 → Lean 4 形式化 → 自主发现 → 推理扩展 → 输出新论文

核心模块：[`article_reasoning.py`](autoforge/engine/article_reasoning.py) — 统一编排 8 个阶段的完整管线

### 形式化验证与定理证明

| 模块 | 功能 | 技术方法 |
|------|------|----------|
| [Lean 4 MCTS 证明搜索](autoforge/engine/provers/proof_search.py) | 策略空间蒙特卡洛树搜索 | HILBERT 递归分解 + COPRA + STP |
| [Lean Lake 集成](autoforge/engine/provers/lean_lake.py) | 真实 Lean 4 编译与 Mathlib 项目管理 | Lake 构建系统，32 组概念→import 映射 |
| [RL 证明搜索](autoforge/engine/rl_proof_search.py) | 强化学习驱动的证明策略 | AlphaProof/DeepSeek-Prover-V2 风格 PUCT-MCTS |
| [多证明器交叉验证](autoforge/engine/provers/multi_prover.py) | Coq、Isabelle、TLA+、Z3/SMT、Dafny | 6 后端并行验证 |
| [密集嵌入检索](autoforge/engine/dense_retrieval.py) | 前提选择替代 Jaccard | ReProver/LeanDojo 风格 + FAISS |
| [证明嵌入迁移](autoforge/engine/proof_embedding.py) | 跨领域策略迁移学习 | 向量记忆库 + 经验追踪 |
| [标准基准评测](autoforge/engine/benchmark_eval.py) | miniF2F / PutnamBench / LeanWorkbook / ProofNet | Pass@k 无偏估计 |

### 自主科研发现

| 模块 | 功能 | 技术方法 |
|------|------|----------|
| [自主定理发现](autoforge/engine/autonomous_discovery.py) | 从论文提取核心 → 生成猜想 → 过滤新颖性 → 评估深度 | DomainContext 模板 + Thompson 采样策略选择 |
| [自对弈猜想生成](autoforge/engine/self_play_conjecture.py) | 双智能体 Conjecturer/Prover 对弈 | STP (ICML 2025) + 贝叶斯难度校准 |
| [推理核心自增长](autoforge/engine/reasoning_extension.py) | 从最小公理核心出发，迭代生成深层结论 | Thompson 采样 + 出版级质量门控 |
| [跨领域科学推理](autoforge/engine/theoretical_reasoning.py) | TheoryGraph + 12 种推理策略 + 多模态验证 | 10 种验证模式加权融合 |
| [结构化世界模型](autoforge/engine/world_model.py) | TheoryGraph 时序查询层 + 跨会话持久化 | Kosmos (2025) |
| [课程学习](autoforge/engine/curriculum_learning.py) | 复杂度排序 + 正迁移追踪 | LeanAgent (ICLR 2025) |

### 论文全流程

| 模块 | 功能 | 技术方法 |
|------|------|----------|
| [闭环实验管线](autoforge/engine/experiment_loop.py) | 假设 → 代码 → 运行 → 分析 → 消融实验 → 迭代 | AI Scientist v2 |
| [自动论文撰写](autoforge/engine/paper_writer.py) | LaTeX 生成 + BibTeX + 图表 + 模板 | NeurIPS/ICML/ICLR/ArXiv 模板 |
| [文献检索与分析](autoforge/engine/literature_search.py) | 引用图谱遍历 + SPECTER2 语义搜索 + 全文分析 + 研究空白检测 | Semantic Scholar API + arXiv |
| [论文复现管线](autoforge/engine/paper_repro.py) | 目标推断 → 信号提取 → 代码生成 → 执行 → 指标比较 → 报告 | OpenReview API 集成 |
| [VLM 图表分析](autoforge/engine/vlm_figure.py) | 图表提取 → 视觉分析 → 数据提取 → 复现 → 验证 | 视觉语言模型 |
| [符号计算后端](autoforge/engine/symbolic_compute.py) | SymPy/SageMath 集成，LaTeX↔SymPy 双向转换 | 代数恒等式验证 + 极限/级数检查 |
| [同行评审模拟](autoforge/engine/peer_review.py) | 多审稿人 + 作者反驳 + 元审稿 + 迭代修改 | 6 种审稿角色 |

### 采用的核心技术与灵感来源

| 技术 | 来源 | 关键创新 |
|------|------|----------|
| **PUCT-MCTS 证明搜索** | AlphaProof (DeepMind, 2024) | 将 AlphaZero 公式适配至策略空间搜索 |
| **递归分解证明** | HILBERT (NeurIPS 2025) | 四组件架构：informal reasoner + prover + verifier + retriever |
| **自对弈猜想** | STP (ICML 2025, 28.5% LeanWorkbook) | 双智能体 + 贝叶斯难度校准锁定 50% 成功率甜区 |
| **密集前提检索** | ReProver / LeanDojo (NeurIPS 2023) | 替代 Jaccard 的密集嵌入 + FAISS 索引 |
| **课程学习** | LeanAgent (ICLR 2025) | 复杂度排序 + 正迁移追踪的终身学习 |
| **结构化世界模型** | Kosmos (2025) | 时序查询 + 跨会话持久化的知识管理 |
| **闭环实验** | AI Scientist v2 (2025) | 假设→代码→执行→分析→消融→迭代的全闭环 |
| **过程奖励模型** | CodePRM (2024) | 步级代码质量评估而非仅结果级 |
| **语言强化学习** | Reflexion (NeurIPS 2023) | 语言化记忆 + 失败模式避免 |
| **文本反向传播** | EvoMAC (2024) | Agent 间自然语言梯度反馈 |
| **自适应算力** | Adaptive TTC (2024) | 难度感知资源分配 + 自校准 |
| **知识图谱自增长** | CapabilityDAG (内部) | 跨项目能力积累，社区可合并 |

---

## 系统要求

- **Python 3.11+** — [python.org](https://python.org)
- **至少一个 LLM API Key** — [Anthropic](https://console.anthropic.com/) / [OpenAI](https://platform.openai.com/api-keys) / [Google](https://aistudio.google.com/apikey)
- **Git**（推荐）— 用于 Worktree 隔离并行开发
- **Docker**（可选）— 用于沙盒执行
- **Lean 4**（可选）— 用于形式化定理证明

---

## 许可证

MIT
