#!/usr/bin/env python3
"""Verify both current Host repositories without importing their runtimes."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_ascend_attention_optimizations.contracts import verify_hosts

if len(sys.argv) != 3:
    raise SystemExit(
        f"usage: {sys.argv[0]} /path/to/vllm-hust /path/to/vllm-ascend-hust"
    )
print(json.dumps(verify_hosts(sys.argv[1], sys.argv[2]), indent=2, sort_keys=True))
