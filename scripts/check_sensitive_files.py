from __future__ import annotations

import fnmatch
import subprocess
import sys


SENSITIVE_PATTERNS = [
    # Local audit docs (must never be committed)
    "AutoForge_Academic_Audit*.docx",
    "AutoForge_Audit*.docx",
    "docs/CLAIMS_AUDIT.md",
    # Local security/sensitive review notes
    "CODE_REVIEW.md",
]


def _git_ls_files() -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"WARN: cannot run git ls-files: {e}")
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _matches(path: str) -> bool:
    p = path.replace("\\", "/")
    for pat in SENSITIVE_PATTERNS:
        if fnmatch.fnmatch(p, pat):
            return True
    return False


def main() -> int:
    tracked = _git_ls_files()
    bad = [p for p in tracked if _matches(p)]
    if bad:
        print("ERROR: sensitive files are tracked by git. Remove them from the index and add to .gitignore.")
        for p in bad:
            print(f"- {p}")
        return 1
    print("OK: no sensitive files tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
