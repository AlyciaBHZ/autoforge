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

[English](docs/README_EN.md) | [开发者文档](CLAUDE.md)

---

## 快速开始

```bash
pip install autoforge                         # 安装
autoforge "带用户登录的待办事项应用"              # 生成项目
```

首次运行会引导配置 API Key（支持 Anthropic / OpenAI / Google 任意一家）。项目输出到 `workspace/` 目录，开箱即用。

<details>
<summary>可选依赖</summary>

```bash
pip install autoforge[openai]     # OpenAI 支持
pip install autoforge[google]     # Google Gemini 支持
pip install autoforge[search]     # Web 搜索能力
pip install autoforge[channels]   # Telegram / Webhook 频道
pip install autoforge[all]        # 全部安装
```

</details>

<details>
<summary>从源码安装（开发者）</summary>

```bash
git clone https://github.com/AlyciaBHZ/autoforge.git
cd autoforge
pip install -e ".[all]"
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
# 生成项目
autoforge "用 Flask + Vue 做一个书店管理系统，带 JWT 认证"
autoforge "SaaS 产品落地页" --budget 3.00

# 管理运行
autoforge --status                # 查看所有项目
autoforge --resume                # 恢复中断的任务

# 守护进程模式（24/7 后台服务）
autoforge daemon start
autoforge queue "支持 Markdown 的博客系统"
autoforge projects
autoforge deploy <project_id>
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

<details>
<summary>高级推理能力</summary>

- **定理证明** — Lean 4 形式化证明，含 MCTS 策略搜索与自动修复
- **多证明器验证** — 支持 Coq、Isabelle、TLA+、Z3/SMT、Dafny 交叉验证
- **跨领域科研推理** — 理论图谱构建、多模态验证、理论演化与论文生成
- **自增长知识图谱** — 跨项目能力积累，社区可合并的通用知识网络
- **提示词自优化** — 自动 A/B 测试与进化式提示词改进
- **跨项目 RAG 检索** — BM25+TF-IDF 混合检索历史项目代码
- **多智能体辩论** — 奖励引导的架构方案多角度辩论
- **安全漏洞扫描** — 模式匹配 + LLM 深度分析双重安全检测
- **工作流自我进化** — 基于历史运行 fitness 的跨项目策略进化

</details>

---

## 系统要求

- **Python 3.11+** — [python.org](https://python.org)
- **至少一个 LLM API Key** — [Anthropic](https://console.anthropic.com/) / [OpenAI](https://platform.openai.com/api-keys) / [Google](https://aistudio.google.com/apikey)
- **Git**（推荐）— 用于 Worktree 隔离并行开发
- **Docker**（可选）— 用于沙盒执行

---

## 许可证

MIT
