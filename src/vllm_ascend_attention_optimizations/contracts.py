"""Current Core/Ascend source identities used by the qualification gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

CORE_REVISION = "c4df37dfcc7f72ca41cd43801a468526ecf99a2b"
ASCEND_REVISION = "7602797593170da4c531f1d8882a7b82dce8a1bc"
CORE_FILES = {
    "vllm/v1/attention/backends/utils.py": "7d35af7b6b9c255c6762eb89702bb1a3ed0d1b292722be46dd6455ecf2410641",
}
ASCEND_FILES = {
    "vllm_ascend/attention/attention_v1.py": "44c43d3610fbc5d50b2a52ef82d5b7ff54b7a1121e5b83b6c9d046f17004c36f",
    "vllm_ascend/attention/utils.py": "20bc2969b3d6ce322b9a74fae1005df6094a8bc9896255a33bbf7d2391d1c460",
    "vllm_ascend/compilation/graph_fusion_pass_manager.py": "1bc376f1ee26aeaf3ebf8fc06667cc347b9074ce4bcfb33635e7574d534adba3",
    "vllm_ascend/ops/triton/batch_invariant/matmul.py": "2c42c1ab8110d4bc17bece46d379cc6c83044a36debd0411efe4bc2d292af20c",
    "vllm_ascend/worker/model_runner_v1.py": "d554280ddaf8ec25043cff9195773a03d16b4f331fb868550ce7842c11d4aa62",
}


class QualificationError(RuntimeError):
    pass


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verify(root: Path, records: dict[str, str]) -> list[dict[str, str]]:
    checks = []
    for relative, expected in records.items():
        actual = digest(root / relative)
        if actual != expected:
            raise QualificationError(
                f"source mismatch for {relative}: expected {expected}, got {actual}"
            )
        checks.append({"path": relative, "sha256": actual})
    return checks


def verify_hosts(core_root: str | Path, ascend_root: str | Path) -> dict:
    return {
        "core_revision": CORE_REVISION,
        "ascend_revision": ASCEND_REVISION,
        "core": _verify(Path(core_root), CORE_FILES),
        "ascend": _verify(Path(ascend_root), ASCEND_FILES),
    }
