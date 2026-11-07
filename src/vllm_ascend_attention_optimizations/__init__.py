"""Default-off qualification gate for recovered attention candidates."""

from __future__ import annotations

import json
import os
from importlib.resources import files

from .contracts import ASCEND_REVISION, CORE_REVISION, QualificationError

_SELECTED = None


def candidate_status() -> dict:
    return json.loads(
        files("vllm_ascend_attention_optimizations")
        .joinpath("candidate-status.json")
        .read_text()
    )


def load() -> None:
    global _SELECTED
    selected = os.getenv("VLLM_HUST_ATTENTION_OPTIMIZATIONS", "").strip()
    if _SELECTED is not None:
        if selected != _SELECTED:
            raise QualificationError("attention selector changed in a live process")
        return
    if not selected:
        _SELECTED = ""
        return
    records = candidate_status()["candidates"]
    if selected not in records:
        raise QualificationError(f"unknown attention optimization: {selected}")
    record = records[selected]
    raise QualificationError(
        f"{selected} is not admitted: {record['status']}: {record['reason']}"
    )


def receipt() -> dict:
    return {
        "core_revision": CORE_REVISION,
        "ascend_revision": ASCEND_REVISION,
        "selected": _SELECTED or "",
        "installed": False,
        "candidates": candidate_status()["candidates"],
    }
