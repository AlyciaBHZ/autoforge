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
    if not project_dir.is_dir():
        return f"# Deploy Guide\n\nError: project directory not found: {project_dir}\n"

    info = detect_framework(project_dir)
    framework = info["framework"]
    name = project_name or project_dir.name
    # Quote the path for safe use in shell commands
    safe_dir = str(project_dir).replace(" ", "\\ ")

    lines = [
        f"# {name} — Vercel Deployment Guide / 部署指南",
        "",
        f"Detected framework / 检测到框架: **{framework.upper()}**",
        "",
        "---",
        "",
        "## Step 1: Push to GitHub / 第 1 步：推送代码到 GitHub",
        "",
        "```bash",
        f"cd {safe_dir}",
        "git init",
        "git add .",
        'git commit -m "Initial commit from AutoForge"',
        "# Create a repo on GitHub, then:",
        f"git remote add origin https://github.com/YOUR_USERNAME/{name}.git",
        "git push -u origin main",
        "```",
        "",
        "## Step 2: Deploy on Vercel / 第 2 步：在 Vercel 部署",
        "",
        "1. Go to [vercel.com](https://vercel.com), sign in with GitHub / 打开 vercel.com 登录",
        "2. Click **Add New Project** / 点击 Add New Project",
        f"3. Select your `{name}` repo / 选择你的仓库",
        "4. Vercel auto-detects framework settings / 自动检测框架配置：",
        f"   - Framework Preset: **{framework}**",
        f"   - Build Command: `{info['build_command']}`",
        f"   - Output Directory: `{info['output_directory']}`",
        f"   - Install Command: `{info['install_command']}`",
        "5. Click **Deploy** / 点击 Deploy",
        "",
    ]

    # Environment variables section
    if info["env_vars"]:
        lines.extend([
            "## Step 3: Environment Variables / 第 3 步：配置环境变量",
            "",
            "Vercel Dashboard → Settings → Environment Variables:",
            "",
            "| Variable | Description | How to Get |",
            "|----------|-------------|------------|",
        ])
        for var in info["env_vars"]:
            hint = _get_env_var_hint(var)
            lines.append(f"| `{var}` | {hint['desc']} | {hint['how']} |")
        lines.extend(["", ""])

    # Domain section
    lines.extend([
        "## Custom Domain / 域名配置（可选）",
        "",
        "Vercel provides free `*.vercel.app` subdomains. For custom domains:",
        "",
        "### Free Option / 免费方案",
        "- Use the default `your-project.vercel.app` — free with HTTPS",
        "",
        "### Custom Domain Registrars / 自定义域名",
        "",
        "| Registrar | Price | Notes |",
        "|-----------|-------|-------|",
        "| **Cloudflare** | ~$9/yr (.com) | At-cost pricing, free DNS |",
        "| **Namecheap** | ~$9-13/yr (.com) | First-year deals, free privacy |",
        "| **Porkbun** | ~$9/yr (.com) | Cheap, free SSL + privacy |",
        "",
        "**Setup:**",
        "1. Buy domain at registrar",
        "2. Vercel Dashboard → Settings → Domains → Add Domain",
        "3. Add CNAME record pointing to `cname.vercel-dns.com`",
        "4. Wait for DNS propagation (minutes to hours)",
        "5. Vercel auto-configures HTTPS",
        "",
        "---",
        "",
        "## Vercel Free Tier / 免费额度",
        "",
        "Vercel Hobby plan (free) includes:",
        "- 100 GB bandwidth/month",
        "- Serverless Functions (100 GB-Hours/month)",
        "- Automatic HTTPS",
        "- Preview Deployments (auto-deploy per PR)",
        "- Edge Network (global CDN)",
        "",
        "More than enough for personal projects and MVPs.",
        "",
        "---",
        "",
        "## Continuous Deployment / 持续部署",
        "",
        "Once connected to GitHub, every `git push` auto-deploys.",
        "Each Pull Request also gets a preview link.",
    ])

    return "\n".join(lines)


def _get_env_var_hint(var: str) -> dict[str, str]:
    """Provide hints for common env var names."""
    var_upper = var.upper()
    hints = {
        "DATABASE_URL": {
            "desc": "Database URL",
            "how": "Vercel Postgres / Supabase / PlanetScale (free tier)",
        },
        "NEXTAUTH_SECRET": {
            "desc": "NextAuth encryption key",
            "how": "Run `openssl rand -base64 32`",
        },
        "NEXTAUTH_URL": {
            "desc": "App URL",
            "how": "Set to `https://your-project.vercel.app`",
        },
        "NEXT_PUBLIC_API_URL": {
            "desc": "Public API URL",
            "how": "Set to your Vercel deployment URL",
        },
        "JWT_SECRET": {
            "desc": "JWT signing key",
            "how": "Run `openssl rand -base64 32`",
        },
        "OPENAI_API_KEY": {
            "desc": "OpenAI API key",
            "how": "[platform.openai.com](https://platform.openai.com)",
        },
        "ANTHROPIC_API_KEY": {
            "desc": "Anthropic API key",
            "how": "[console.anthropic.com](https://console.anthropic.com)",
        },
        "STRIPE_SECRET_KEY": {
            "desc": "Stripe secret key",
            "how": "[dashboard.stripe.com](https://dashboard.stripe.com)",
        },
        "STRIPE_PUBLISHABLE_KEY": {
            "desc": "Stripe publishable key",
            "how": "[dashboard.stripe.com](https://dashboard.stripe.com)",
        },
        "REDIS_URL": {
            "desc": "Redis connection URL",
            "how": "Vercel KV / Upstash (free tier)",
        },
        "S3_BUCKET": {
            "desc": "S3 bucket name",
            "how": "AWS S3 or Cloudflare R2 (free 10 GB)",
        },
    }

    if var_upper in hints:
        return hints[var_upper]

    # Generic hints based on patterns
    if "SECRET" in var_upper or "KEY" in var_upper:
        return {"desc": "Secret / credential", "how": "See service docs to generate or obtain"}
    if "URL" in var_upper:
        return {"desc": "Service URL", "how": "Set to your deployment address"}
    if "TOKEN" in var_upper:
        return {"desc": "Access token", "how": "Obtain from the service platform"}

    return {"desc": "Environment variable", "how": "See project docs"}
