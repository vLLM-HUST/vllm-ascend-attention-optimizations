import importlib

import pytest


def fresh_module():
    import vllm_ascend_attention_optimizations as module

    return importlib.reload(module)


def test_default_is_inert(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_ATTENTION_OPTIMIZATIONS", raising=False)
    module = fresh_module()
    module.load()
    assert module.receipt()["installed"] is False
    assert module.receipt()["selected"] == ""


@pytest.mark.parametrize(
    "selector,status",
    [
        ("core-boundary", "retired_absorbed"),
        ("persistent-matmul-empty", "retired_not_consumed"),
        ("linear-swiglu-graph", "relocated_operator_family"),
        ("pyramidkv", "port_required"),
    ],
)
def test_legacy_selector_fails_closed(monkeypatch, selector, status):
    monkeypatch.setenv("VLLM_HUST_ATTENTION_OPTIMIZATIONS", selector)
    module = fresh_module()
    with pytest.raises(module.QualificationError, match=status):
        module.load()


def test_unknown_selector_fails_closed(monkeypatch):
    monkeypatch.setenv("VLLM_HUST_ATTENTION_OPTIMIZATIONS", "unknown")
    module = fresh_module()
    with pytest.raises(module.QualificationError, match="unknown"):
        module.load()
