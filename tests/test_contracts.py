import os
from pathlib import Path

import pytest

from vllm_ascend_attention_optimizations.contracts import (
    QualificationError,
    verify_hosts,
)


def test_current_hosts_if_requested():
    core = os.getenv("VLLM_CORE_HOST_ROOT")
    ascend = os.getenv("VLLM_ASCEND_HOST_ROOT")
    if not core or not ascend:
        pytest.skip("set both Host roots for source verification")
    result = verify_hosts(core, ascend)
    assert len(result["core"]) == 1
    assert len(result["ascend"]) == 5


def test_changed_host_fails_closed(tmp_path: Path):
    core = tmp_path / "core"
    ascend = tmp_path / "ascend"
    for path in (
        core / "vllm/v1/attention/backends/utils.py",
        ascend / "vllm_ascend/attention/attention_v1.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
    with pytest.raises(QualificationError):
        verify_hosts(core, ascend)
