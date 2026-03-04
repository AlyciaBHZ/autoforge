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
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![Smoke Tests](https://img.shields.io/badge/tests-31%20checks-brightgreen.svg)](../tests/smoke_test.py)

[English](../README.md) | [开发者文档](../CLAUDE.md)

---

## 快速开始

```bash
./setup.sh                                    # 安装依赖
export ANTHROPIC_API_KEY="sk-..."             # 设置任意支持的 LLM Key
python forge.py "带用户登录的待办事项应用"       # 生成项目
```

项目输出到 `workspace/` 目录，开箱即用。

---

## 它能做什么

AutoForge 通过 6 个专业化 AI Agent 协作，经过 5 阶段流水线，将一句自然语言描述转化为完整的代码项目。需求分析、架构设计、并行代码生成、自动化测试、代码审查、重构优化、部署打包，全程自动完成。

**核心特性：**
- 全栈 Web 应用、API 服务、CLI 工具、移动端应用，一行命令搞定
- 多 LLM 提供商支持，可跨厂商混搭（Opus 做规划，Flash 写代码）
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

| Agent | 角色 | 模型层级 |
|-------|------|----------|
| **Director** | 需求分析与范围界定 | Strong (Opus) |
| **Architect** | 系统设计与任务 DAG | Strong (Opus) |
| **Builder** | 代码生成（可并行） | Fast (Sonnet) |
| **Reviewer** | 代码审查与评分 | Fast (Sonnet) |
| **Tester** | 构建、测试、自动修复循环 | Fast (Sonnet) |
| **Gardener** | 重构与安全修复 | Fast (Sonnet) |

---

## 学术基础

AutoForge 集成了近年 AI 与软件工程领域的前沿研究成果：

| 引擎 | 源文件 | 论文 / 参考 |
|------|--------|-------------|
| **RethinkMCTS** | `search_tree.py` | RethinkMCTS (2024) -- 基于执行反馈的 MCTS 思维链修正 |
| **EvoMAC** | `evomac.py` | EvoMAC (ICLR 2025) -- 自然语言梯度的文本反向传播 |
| **SICA** | `sica.py` | SICA (ICLR 2025 Workshop) + STO (NeurIPS 2025) -- 自改进编码智能体 |
| **Reflexion** | `reflexion.py` | Reflexion (NeurIPS 2023, Shinn et al.) -- 语言强化学习 |
| **CodePRM** | `process_reward.py` | CodePRM (ACL 2025) -- 代码生成步骤级过程奖励模型 |
| **LDB** | `ldb_debugger.py` | LDB (ACL 2024, Zhong et al.) -- 块级故障定位 |
| **Adaptive Compute** | `adaptive_compute.py` | Scaling LLM Test-Time Compute (ICLR 2025) -- 难度自适应资源分配 |
| **Speculative Pipeline** | `speculative_pipeline.py` | Speculative Actions + Sherlock -- 阶段重叠预执行加速 |
| **Parsel** | `hierarchical_decomp.py` | Parsel (NeurIPS 2023) + CodePlan (ACM 2024) -- 函数级任务分解 |
| **Lean Prover** | `lean_prover.py` | Hilbert (NeurIPS 2025) + COPRA (COLM 2024) + DeepSeek-Prover (ICLR 2025) + STP (ICML 2025) |
| **CapabilityDAG** | `capability_dag.py` | Voyager (NeurIPS 2023) + FunSearch (Nature 2024) -- 自增长知识图谱 |
| **TheoryGraph** | `theoretical_reasoning.py` | 跨领域科研推理与多模态验证 |

<details>
<summary>更多集成技术</summary>

- **DSPy/OPRO 提示词优化** (`prompt_optimizer.py`) -- 自动提示词自改进
- **跨项目 RAG 检索** (`rag_retrieval.py`) -- BM25+TF-IDF 混合代码检索
- **形式化验证** (`formal_verify.py`) -- 多层次静态分析、类型检查、LLM 形式化分析
- **条件 Agent 辩论** (`agent_debate.py`) -- 奖励引导的多智能体架构辩论 (ICLR 2025)
- **RedCode 安全扫描** (`security_scan.py`) -- 模式匹配 + LLM 漏洞深度分析 (NeurIPS 2024)
- **进化引擎** (`evolution.py`) -- MAP-Elites 跨项目工作流自我进化

</details>

---

## 支持的 LLM 提供商

| 提供商 | 认证方式 | 推荐模型 |
|--------|----------|----------|
| **Anthropic** | `ANTHROPIC_API_KEY` 或 `~/.autoforge/config.toml` | Claude Opus 4（强）、Sonnet 4.5（快） |
| **OpenAI** | `OPENAI_API_KEY` 或 `~/.autoforge/config.toml` | GPT-4o、o3（强）、GPT-4o-mini（快） |
| **Google** | `GOOGLE_API_KEY` 或 `~/.autoforge/config.toml` | Gemini 2.5 Pro（强）、Gemini 2.5 Flash（快） |

支持跨提供商混搭：

```bash
export FORGE_MODEL_STRONG=claude-opus-4-6
export FORGE_MODEL_FAST=gemini-2.5-flash
```

---

## 使用方法

```bash
# 生成项目
python forge.py "用 Flask + Vue 做一个书店管理系统，带 JWT 认证"
python forge.py "SaaS 产品落地页" --budget 3.00

# 管理运行
python forge.py --status          # 查看所有项目
python forge.py --resume          # 恢复中断的任务

# 守护进程模式（24/7 后台服务）
python forge.py daemon start
python forge.py queue "支持 Markdown 的博客系统"
python forge.py projects
python forge.py deploy <project_id>
```

<details>
<summary>成本估算</summary>

| 复杂度 | 示例 | 预估成本 |
|--------|------|:--------:|
| 简单 | Todo App、落地页 | $2--3 |
| 中等 | 博客系统、预约平台 | $4--6 |
| 复杂 | 电商 MVP、多角色平台 | $7--10 |

默认预算上限 $10，可通过 `--budget` 覆盖。

</details>

---

## 系统要求

- **Python 3.11+** -- [python.org](https://python.org)
- **至少一个 LLM API Key** -- [Anthropic](https://console.anthropic.com/) / [OpenAI](https://platform.openai.com/api-keys) / [Google](https://aistudio.google.com/apikey)
- **Git**（推荐）-- 用于 Worktree 隔离并行开发
- **Docker**（可选）-- 用于沙盒执行

---

## 许可证

MIT
