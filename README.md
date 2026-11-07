# vLLM Ascend Attention Optimizations

Default-off Ascend attention-path optimization MOD family for vLLM-HUST. The
current package is a qualification gate: it is inert by default and rejects
all recovered legacy selectors until a current-Host implementation exists.

## Scope

Shared-prefix attention, FIA/FA execution, RoPE and attention-path optimization experiments.

This repository is the canonical integration and evidence home for the family. Every optimization is opt-in and disabled by default until its own compatibility and performance qualification is complete.

## Recovered candidate disposition

| Legacy selector | Disposition | Reason |
| --- | --- | --- |
| `core-boundary` | Retired | Current Core and Ascend implement their own evolved decode/prefill boundary logic |
| `persistent-matmul-empty` | Retired | The original formal environment never consumed this path |
| `linear-swiglu-graph` | Relocated | This is an operator/fusion concern and must not be duplicated in the attention family |
| `pyramidkv` | Port required | It depends on the removed `native_model_runner_v1.py` lifecycle and old provider build |

An empty `VLLM_HUST_ATTENTION_OPTIMIZATIONS` value is a no-op. Supplying any
legacy selector fails closed with its disposition. This prevents an old wheel
or deployment preset from silently patching a changed Host.

## Current source audit

The audit is pinned to the Core commit verified by the Ascend Host,
`c4df37dfcc7f72ca41cd43801a468526ecf99a2b`,
and Ascend `7602797593170da4c531f1d8882a7b82dce8a1bc`:

```bash
python tools/check_host_contract.py /path/to/vllm-hust /path/to/vllm-ascend-hust
pytest -q
```

The family remains `retest_pending`. No legacy performance result is eligible
for the website until a new mechanism passes current-Host NPU qualification.

## Ownership

- Maintainer: [ShuhaoZhangTony](https://github.com/ShuhaoZhangTony)
- Advisors: none recorded
- Organization: [vLLM-HUST](https://github.com/vLLM-HUST)

## Evidence

Measured single-MOD, composition, and compatibility results are published through [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark) and presented with the corresponding MOD in the [MOD catalog](https://vllm-hust.sage.org.ai/plugins.html).
