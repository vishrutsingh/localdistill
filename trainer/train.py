#!/usr/bin/env python3
"""
train.py - DEPRECATED: Use distill.py run instead.

This is a thin compatibility wrapper that delegates to the main orchestrator.
It exists only so Docker/trainer Dockerfile doesn't break.

Recommended:
    python distill.py run --mode preference
"""

import sys
import subprocess


def main():
    print("=" * 60)
    print("⚠️  trainer/train.py is deprecated.")
    print("   Use: python distill.py run [OPTIONS]")
    print("=" * 60)
    print()

    # Forward to distill.py
    cmd = [sys.executable, "distill.py", "run"] + sys.argv[1:]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
