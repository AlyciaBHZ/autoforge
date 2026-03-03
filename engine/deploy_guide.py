"""Deploy Guide — generates Vercel deployment instructions for completed projects.

Detects the framework from the generated project and produces a step-by-step
guide including environment variables, domain setup, and cost-effective options.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def detect_framework(project_dir: Path) -> dict[str, Any]:
    """Detect the project's framework and deployment characteristics."""
    info: dict[str, Any] = {
        "framework": "unknown",
        "build_command": "npm run build",
        "output_directory": ".next",
        "install_command": "npm install",
        "dev_command": "npm run dev",
        "env_vars": [],
        "node_version": "20",
    }

    package_json = project_dir / "package.json"
    if package_json.exists():
        try:
            pkg = json.loads(package_json.read_text(encoding="utf-8"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

            if "next" in deps:
                info["framework"] = "nextjs"
                info["output_directory"] = ".next"
            elif "nuxt" in deps or "nuxt3" in deps:
                info["framework"] = "nuxt"
                info["output_directory"] = ".output"
            elif "vite" in deps or "@vitejs/plugin-react" in deps:
                info["framework"] = "vite"
                info["output_directory"] = "dist"
                info["build_command"] = "npm run build"
            elif "react-scripts" in deps:
                info["framework"] = "create-react-app"
                info["output_directory"] = "build"
            elif "svelte" in deps or "@sveltejs/kit" in deps:
                info["framework"] = "sveltekit"
                info["output_directory"] = ".svelte-kit"
            elif "astro" in deps:
                info["framework"] = "astro"
                info["output_directory"] = "dist"

            scripts = pkg.get("scripts", {})
            if "build" in scripts:
                info["build_command"] = "npm run build"
            if "dev" in scripts:
                info["dev_command"] = "npm run dev"
        except (json.JSONDecodeError, KeyError):
            pass

    # Detect env vars from .env.example or .env.local.example
    for env_file_name in [".env.example", ".env.local.example", ".env.template"]:
        env_file = project_dir / env_file_name
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=", 1)[0].strip()
                    info["env_vars"].append(key)

    return info


def generate_deploy_guide(project_dir: Path, project_name: str = "") -> str:
    """Generate a Vercel deployment guide for the project."""
    info = detect_framework(project_dir)
    framework = info["framework"]
    name = project_name or project_dir.name

    lines = [
        f"# {name} — Vercel 部署指南",
        "",
        f"检测到框架: **{framework.upper()}**",
        "",
        "---",
        "",
        "## 第 1 步：推送代码到 GitHub",
        "",
        "```bash",
        f"cd {project_dir}",
        "git init",
        "git add .",
        'git commit -m "Initial commit from AutoForge"',
        "# 在 GitHub 创建仓库后：",
        f"git remote add origin https://github.com/YOUR_USERNAME/{name}.git",
        "git push -u origin main",
        "```",
        "",
        "## 第 2 步：在 Vercel 部署",
        "",
        "1. 打开 [vercel.com](https://vercel.com)，用 GitHub 账号登录",
        "2. 点击 **Add New Project**",
        f"3. 选择你的 `{name}` 仓库",
        "4. Vercel 会自动检测框架配置：",
        f"   - Framework Preset: **{framework}**",
        f"   - Build Command: `{info['build_command']}`",
        f"   - Output Directory: `{info['output_directory']}`",
        f"   - Install Command: `{info['install_command']}`",
        "5. 点击 **Deploy**",
        "",
    ]

    # Environment variables section
    if info["env_vars"]:
        lines.extend([
            "## 第 3 步：配置环境变量",
            "",
            "在 Vercel Dashboard → Settings → Environment Variables 中添加：",
            "",
            "| 变量名 | 说明 | 如何获取 |",
            "|--------|------|---------|",
        ])
        for var in info["env_vars"]:
            hint = _get_env_var_hint(var)
            lines.append(f"| `{var}` | {hint['desc']} | {hint['how']} |")
        lines.extend(["", ""])

    # Domain section
    lines.extend([
        "## 域名配置（可选）",
        "",
        "Vercel 免费提供 `*.vercel.app` 子域名。如果你想用自定义域名：",
        "",
        "### 免费方案",
        "- 直接使用 Vercel 默认分配的 `your-project.vercel.app`",
        "- 完全免费，自带 HTTPS",
        "",
        "### 自定义域名（推荐性价比方案）",
        "",
        "| 平台 | 价格 | 特点 |",
        "|------|------|------|",
        "| **Cloudflare Registrar** | ~$9/年 (.com) | 成本价，无加价，免费 DNS |",
        "| **Namecheap** | ~$9-13/年 (.com) | 首年优惠多，免费隐私保护 |",
        "| **Google Domains** | ~$12/年 (.com) | 简洁，自带隐私保护 |",
        "| **Porkbun** | ~$9/年 (.com) | 便宜，免费 SSL + 隐私 |",
        "",
        "**配置步骤：**",
        "1. 在域名注册商购买域名",
        "2. Vercel Dashboard → Settings → Domains → Add Domain",
        "3. 按提示在域名注册商添加 DNS 记录（CNAME 指向 `cname.vercel-dns.com`）",
        "4. 等待 DNS 生效（通常几分钟到几小时）",
        "5. Vercel 自动配置 HTTPS",
        "",
        "---",
        "",
        "## Vercel 免费额度",
        "",
        "Vercel Hobby 计划（免费）包含：",
        "- 100 GB 带宽/月",
        "- Serverless Functions（每月 100 GB-Hours）",
        "- 自动 HTTPS",
        "- Preview Deployments（每个 PR 自动部署预览）",
        "- 边缘网络（全球 CDN）",
        "",
        "对于大多数个人项目和 MVP 来说，免费额度绰绰有余。",
        "",
        "---",
        "",
        "## 持续部署",
        "",
        "连接 GitHub 后，每次 `git push` Vercel 都会自动重新部署。",
        "每个 Pull Request 也会自动生成预览链接。",
    ])

    return "\n".join(lines)


def _get_env_var_hint(var: str) -> dict[str, str]:
    """Provide hints for common env var names."""
    var_upper = var.upper()
    hints = {
        "DATABASE_URL": {
            "desc": "数据库连接地址",
            "how": "Vercel Postgres / Supabase / PlanetScale 免费套餐",
        },
        "NEXTAUTH_SECRET": {
            "desc": "NextAuth 加密密钥",
            "how": "运行 `openssl rand -base64 32` 生成",
        },
        "NEXTAUTH_URL": {
            "desc": "应用 URL",
            "how": "设为 `https://your-project.vercel.app`",
        },
        "NEXT_PUBLIC_API_URL": {
            "desc": "公开 API 地址",
            "how": "设为你的 Vercel 部署 URL",
        },
        "JWT_SECRET": {
            "desc": "JWT 签名密钥",
            "how": "运行 `openssl rand -base64 32` 生成",
        },
        "OPENAI_API_KEY": {
            "desc": "OpenAI API 密钥",
            "how": "[platform.openai.com](https://platform.openai.com) 获取",
        },
        "ANTHROPIC_API_KEY": {
            "desc": "Anthropic API 密钥",
            "how": "[console.anthropic.com](https://console.anthropic.com) 获取",
        },
        "STRIPE_SECRET_KEY": {
            "desc": "Stripe 支付密钥",
            "how": "[dashboard.stripe.com](https://dashboard.stripe.com) 获取",
        },
        "STRIPE_PUBLISHABLE_KEY": {
            "desc": "Stripe 公钥",
            "how": "[dashboard.stripe.com](https://dashboard.stripe.com) 获取",
        },
        "REDIS_URL": {
            "desc": "Redis 连接地址",
            "how": "Vercel KV / Upstash 免费套餐",
        },
        "S3_BUCKET": {
            "desc": "S3 存储桶",
            "how": "AWS S3 或 Cloudflare R2（免费 10 GB）",
        },
    }

    if var_upper in hints:
        return hints[var_upper]

    # Generic hints based on patterns
    if "SECRET" in var_upper or "KEY" in var_upper:
        return {"desc": "密钥/凭证", "how": "按对应服务文档生成或获取"}
    if "URL" in var_upper:
        return {"desc": "服务 URL", "how": "按你的部署地址填写"}
    if "TOKEN" in var_upper:
        return {"desc": "访问令牌", "how": "从对应平台获取"}

    return {"desc": "环境变量", "how": "按项目文档填写"}
