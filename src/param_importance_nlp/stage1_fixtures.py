"""S1.3 deterministic fixtures and independent offline oracles.

The fixture set in this module is deliberately small and fully serialisable.  It
does not import the training loop and its oracle calculations do not call the
production estimator kernels.  The resulting bundle can therefore be copied to
another machine and recomputed without sampling again.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .atomic import sha256_file
from .contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from .core.oracles import (
    ConstantGradientFixture,
    QuadraticLossFixture,
    ZeroMeanNoiseFixture,
    central_difference_gradient,
    compare_tensor_maps_fp64,
)
from .core.quadrature import (
    PathSpec,
    gauss_legendre_rule,
    integrate_path,
    left_rule,
    midpoint_rule,
    right_rule,
    simpson_rule,
    trapezoid_rule,
)
from .core.registry import ParameterRegistry
from .core.tensors import TensorMap


FIXTURE_SCHEMA = "stage1-fixture-manifest-v1"
ORACLE_SCHEMA = "stage1-oracle-bundle-v1"
VALIDATION_SCHEMA = "stage1-oracle-validation-report-v1"
FIXTURE_SET_ID = "stage1-s13-deterministic-v1"
PYTHIA_STEP0_REVISION = "56079904bb80b7f36d3b794089f146e7a4d6efae"


class Stage1FixtureError(ValueError):
    """A committed S1.3 fixture or its oracle bundle is invalid."""


def _as_float_tensor(value: object, *, field: str) -> torch.Tensor:
    try:
        tensor = torch.tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise Stage1FixtureError(f"{field} 不是有效的有限数值数组") from error
    if not bool(torch.isfinite(tensor).all()):
        raise Stage1FixtureError(f"{field} 含 NaN/Inf")
    return tensor


def _tensor_to_wire(value: torch.Tensor) -> dict[str, JSONValue]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return {
        "dtype": "torch.float64",
        "shape": [int(size) for size in tensor.shape],
        "values": [float(item) for item in tensor.reshape(-1).tolist()],
    }


def _tensor_from_wire(value: object, *, field: str) -> torch.Tensor:
    if not isinstance(value, Mapping):
        raise Stage1FixtureError(f"{field} 必须是 tensor wire object")
    shape = value.get("shape")
    values = value.get("values")
    if (
        not isinstance(shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        or not isinstance(values, list)
    ):
        raise Stage1FixtureError(f"{field} 的 shape/values 非法")
    tensor = _as_float_tensor(values, field=f"{field}.values")
    expected = math.prod(shape) if shape else 1
    if tensor.numel() != expected:
        raise Stage1FixtureError(f"{field} 的 values 数量与 shape 不一致")
    return tensor.reshape(tuple(shape))


def _tensor_map_to_wire(value: TensorMap) -> dict[str, JSONValue]:
    return {name: _tensor_to_wire(tensor) for name, tensor in value.items()}


def _tensor_map_from_wire(value: object, *, field: str, registry: ParameterRegistry | None = None) -> TensorMap:
    if not isinstance(value, Mapping) or not value:
        raise Stage1FixtureError(f"{field} 必须是非空 tensor map")
    return TensorMap(
        {
            str(name): _tensor_from_wire(item, field=f"{field}.{name}")
            for name, item in value.items()
        },
        registry=registry,
    )


def _nested_tensor_map(value: object, *, field: str) -> TensorMap:
    if not isinstance(value, Mapping) or not value:
        raise Stage1FixtureError(f"{field} 必须是非空参数数组")
    tensors = {
        str(name): _as_float_tensor(item, field=f"{field}.{name}")
        for name, item in value.items()
    }
    return TensorMap(tensors)


def _map_to_nested(value: TensorMap) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for name, tensor in value.items():
        result[name] = tensor.detach().to(device="cpu", dtype=torch.float64).tolist()
    return result


def _wire_hash(value: object) -> str:
    return canonical_json_hash(value)


def _manifest_without_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("manifest_hash", None)
    return body


def _load_manifest(source_root: str | Path) -> dict[str, Any]:
    path = Path(source_root) / "fixtures/stage1/stage1-s13-v1.json"
    try:
        raw = load_canonical_json(path)
    except Exception as error:
        raise Stage1FixtureError(f"无法读取 S1.3 committed manifest: {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != FIXTURE_SCHEMA:
        raise Stage1FixtureError("S1.3 fixture manifest schema_version 错误")
    if raw.get("fixture_set_id") != FIXTURE_SET_ID:
        raise Stage1FixtureError("S1.3 fixture_set_id 错误")
    declared_hash = raw.get("manifest_hash")
    if not isinstance(declared_hash, str) or declared_hash != _wire_hash(_manifest_without_hash(raw)):
        raise Stage1FixtureError("S1.3 fixture manifest hash 不匹配")
    required = {"gradient_matrix", "analytic", "tiny_transformer", "noise", "pythia_14m"}
    if set(raw) - {
        "schema_version",
        "fixture_set_id",
        "generator_version",
        "seed_plan",
        "gradient_matrix",
        "analytic",
        "tiny_transformer",
        "noise",
        "pythia_14m",
        "manifest_hash",
    }:
        raise Stage1FixtureError("S1.3 fixture manifest 含未知顶层字段")
    if not required.issubset(raw):
        raise Stage1FixtureError("S1.3 fixture manifest 缺少 fixture 层级")
    for section_name in required:
        section = raw[section_name]
        if not isinstance(section, (dict, list)):
            raise Stage1FixtureError(f"S1.3 manifest section {section_name} 类型错误")
    return raw


def _git_source_hashes(source_root: Path) -> dict[str, JSONValue]:
    paths = (
        "fixtures/stage1/stage1-s13-v1.json",
        "plan/stage1/03_fixtures_and_oracles.md",
        "schemas/stage1/fixture-manifest-v1.json",
        "schemas/stage1/oracle-bundle-v1.json",
        "schemas/stage1/oracle-validation-report-v1.json",
        "schemas/stage1/s1-3-formalization-index-v1.json",
        "schemas/stage1/s1-3-validation-v1.json",
        "src/param_importance_nlp/stage1_fixtures.py",
        "src/param_importance_nlp/experiments/stage01_task_runners.py",
        "tests/test_stage1_s13_fixtures.py",
        "ops/stage1/formalize_s1_3.py",
    )
    result: dict[str, JSONValue] = {}
    for relative in paths:
        path = source_root / relative
        if path.is_file():
            result[relative] = sha256_file(path)
    return result


def _explicit_mean(samples: Sequence[TensorMap]) -> TensorMap:
    if not samples:
        raise Stage1FixtureError("explicit mean 不能处理空样本")
    reference = samples[0]
    reference.assert_finite()
    totals = {name: torch.zeros_like(tensor, dtype=torch.float64) for name, tensor in reference.items()}
    for index, sample in enumerate(samples):
        reference.assert_compatible(sample)
        sample.assert_finite()
        for name, tensor in sample.items():
            totals[name].add_(tensor.to(dtype=torch.float64, device="cpu"))
    return TensorMap({name: value / float(len(samples)) for name, value in totals.items()})


def _explicit_raw(samples: Sequence[TensorMap]) -> TensorMap:
    """Compute raw ``(mean gradient)^2`` with explicit FP64 arithmetic."""

    if not samples:
        raise Stage1FixtureError("explicit raw oracle 不能处理空样本")
    return _explicit_mean(samples).map(torch.square)


def _explicit_pair_oracle(
    samples: Sequence[TensorMap],
    weights: Sequence[float] | None = None,
) -> TensorMap:
    if len(samples) < 2:
        raise Stage1FixtureError("explicit ordered-pair oracle 要求 M>=2")
    reference = samples[0]
    reference.assert_finite()
    normalized = [1.0] * len(samples) if weights is None else [float(value) for value in weights]
    if len(normalized) != len(samples) or any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise Stage1FixtureError("explicit ordered-pair oracle 权重非法")
    total = {name: torch.zeros_like(tensor, dtype=torch.float64) for name, tensor in reference.items()}
    denominator = 0.0
    for left_index, left in enumerate(samples):
        reference.assert_compatible(left)
        for right_index, right in enumerate(samples):
            reference.assert_compatible(right)
            if left_index == right_index:
                continue
            pair_weight = normalized[left_index] * normalized[right_index]
            denominator += pair_weight
            for name in total:
                total[name].add_(
                    left[name].to(dtype=torch.float64, device="cpu")
                    * right[name].to(dtype=torch.float64, device="cpu"),
                    alpha=pair_weight,
                )
    if denominator <= 0 or not math.isfinite(denominator):
        raise Stage1FixtureError("explicit ordered-pair oracle 分母非法")
    return TensorMap({name: value / denominator for name, value in total.items()})


class TinyCausalTransformer(nn.Module):
    """A deterministic, dropout-free causal transformer used only by S1.3."""

    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attention_out = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.mlp_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        query, key, value = self.attention_qkv(hidden).chunk(3, dim=-1)
        scale = float(query.shape[-1]) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        length = input_ids.shape[-1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        hidden = self.layer_norm(hidden + self.attention_out(attended))
        feed_forward = self.mlp_fc2(torch.tanh(self.mlp_fc1(hidden)))
        hidden = self.final_norm(hidden + feed_forward)
        return self.lm_head(hidden)


def build_tiny_transformer(
    spec: Mapping[str, Any],
) -> tuple[TinyCausalTransformer, ParameterRegistry, dict[str, torch.Tensor]]:
    seed = spec.get("seed")
    vocab_size = spec.get("vocab_size")
    hidden_size = spec.get("hidden_size")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (seed, vocab_size, hidden_size)):
        raise Stage1FixtureError("Tiny Transformer seed/vocab/hidden 配置非法")
    model = TinyCausalTransformer(vocab_size=vocab_size, hidden_size=hidden_size).to(
        device="cpu", dtype=torch.float64
    )
    # PyTorch's seeded initializer can change across releases.  The fixture
    # therefore uses the frozen seed as an input to a small explicit arithmetic
    # initializer, making the initial state stable across the local and server
    # runtimes without consuming any global RNG state.
    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(model.named_parameters()):
            positions = torch.arange(parameter.numel(), dtype=torch.float64, device="cpu")
            values = ((positions + float(seed + 17 * parameter_index)) % 29.0 - 14.0) / 10.0
            if name.endswith("layer_norm.weight") or name.endswith("final_norm.weight"):
                values = 1.0 + values / 20.0
            parameter.copy_(values.reshape_as(parameter))
    optimizer = torch.optim.SGD(model.parameters(), lr=float(spec.get("learning_rate", 0.05)), foreach=False)
    registry = ParameterRegistry.from_model(model, optimizer)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    if tuple(initial) != registry.eligible_names:
        initial = {name: registry.parameter(name).detach().clone() for name in registry.eligible_names}
    return model, registry, initial


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_hash(
        {
            name: _tensor_to_wire(state[name])
            for name in sorted(state)
        }
    )


def _tiny_inputs(spec: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[tuple[int, ...], ...]]:
    tokens = spec.get("tokens")
    masks = spec.get("target_masks")
    sample_ids = spec.get("sample_ids")
    groups = spec.get("microbatch_groups")
    if not isinstance(tokens, list) or not isinstance(masks, list) or not isinstance(sample_ids, list) or not isinstance(groups, list):
        raise Stage1FixtureError("Tiny Transformer 输入字段类型错误")
    if len(tokens) == 0 or len(tokens) != len(masks) or len(tokens) != len(sample_ids):
        raise Stage1FixtureError("Tiny Transformer 输入样本数量不一致")
    token_tensor = torch.tensor(tokens, dtype=torch.long, device="cpu")
    mask_tensor = torch.tensor(masks, dtype=torch.float64, device="cpu")
    if token_tensor.ndim != 2 or mask_tensor.shape != (token_tensor.shape[0], token_tensor.shape[1] - 1):
        raise Stage1FixtureError("Tiny Transformer token/mask shape 错误")
    if not bool(((token_tensor >= 0) & (token_tensor < int(spec["vocab_size"]))).all()):
        raise Stage1FixtureError("Tiny Transformer token 超出 vocab")
    if not bool(((mask_tensor == 0) | (mask_tensor == 1)).all()):
        raise Stage1FixtureError("Tiny Transformer target mask 必须为 0/1")
    if any(not isinstance(item, str) or not item for item in sample_ids):
        raise Stage1FixtureError("Tiny Transformer sample_id 非法")
    normalized_groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group in groups:
        if not isinstance(group, list) or not group:
            raise Stage1FixtureError("Tiny Transformer microbatch group 不能为空")
        normalized = tuple(int(item) for item in group)
        if any(item < 0 or item >= len(tokens) or item in seen for item in normalized):
            raise Stage1FixtureError("Tiny Transformer microbatch group 不覆盖恰好一次")
        seen.update(normalized)
        normalized_groups.append(normalized)
    if seen != set(range(len(tokens))):
        raise Stage1FixtureError("Tiny Transformer microbatch group 未覆盖全部样本")
    return token_tensor, mask_tensor, tuple(sample_ids), tuple(normalized_groups)


def _tiny_gradient(
    model: TinyCausalTransformer,
    registry: ParameterRegistry,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    indices: Sequence[int],
) -> tuple[float, int, TensorMap]:
    model.zero_grad(set_to_none=True)
    selected = torch.tensor(list(indices), dtype=torch.long, device="cpu")
    input_ids = tokens.index_select(0, selected)
    target_ids = input_ids[:, 1:]
    target_mask = masks.index_select(0, selected)
    logits = model(input_ids[:, :-1])
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_mask)
    valid = int(target_mask.sum().item())
    if valid <= 0:
        raise Stage1FixtureError("Tiny Transformer microbatch 没有有效 target token")
    loss = (token_loss * target_mask).sum() / float(valid)
    loss.backward()
    gradients: dict[str, torch.Tensor] = {}
    for name in registry.eligible_names:
        gradient = registry.parameter(name).grad
        if gradient is None:
            raise Stage1FixtureError(f"Tiny Transformer 参数 {name} 缺少梯度")
        gradients[name] = gradient.detach().clone().to(dtype=torch.float64, device="cpu")
    return float(loss.detach().item()), valid, TensorMap(gradients, registry=registry)


def _reset_model(model: nn.Module, initial: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(initial[name])
        model.zero_grad(set_to_none=True)


def _compare(actual: TensorMap, expected: TensorMap, *, natural_scale: float = 1.0) -> dict[str, JSONValue]:
    return compare_tensor_maps_fp64(
        actual,
        expected,
        natural_scale=natural_scale,
        atol=1e-11,
        rtol=1e-9,
        normalized_l2_limit=1e-9,
    ).to_dict()


def _check(check_id: str, comparison: Mapping[str, Any] | bool, *, detail: str = "") -> dict[str, JSONValue]:
    passed = bool(comparison if isinstance(comparison, bool) else comparison.get("passed", False))
    result: dict[str, JSONValue] = {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
    }
    if detail:
        result["detail"] = detail
    if not isinstance(comparison, bool):
        result["comparison"] = dict(comparison)
    return result


def _gradient_cases(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    section = manifest["gradient_matrix"]
    if not isinstance(section, Mapping):
        raise Stage1FixtureError("gradient_matrix section 类型错误")
    shapes_raw = section.get("coordinate_shapes")
    cases_raw = section.get("cases")
    if not isinstance(shapes_raw, Mapping) or not isinstance(cases_raw, list) or not cases_raw:
        raise Stage1FixtureError("gradient_matrix 缺少 coordinate_shapes/cases")
    shapes = {
        str(name): tuple(int(size) for size in shape)
        for name, shape in shapes_raw.items()
        if isinstance(shape, list)
    }
    if not shapes or any(any(size <= 0 for size in shape) for shape in shapes.values()):
        raise Stage1FixtureError("gradient_matrix coordinate shape 非法")
    cases: dict[str, Any] = {}
    for raw_case in cases_raw:
        if not isinstance(raw_case, Mapping) or not isinstance(raw_case.get("case_id"), str):
            raise Stage1FixtureError("gradient_matrix case 标识非法")
        case_id = str(raw_case["case_id"])
        samples = raw_case.get("samples")
        if not isinstance(samples, list) or not samples:
            raise Stage1FixtureError(f"gradient_matrix case {case_id} 样本为空")
        parsed: list[TensorMap] = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or set(sample) != set(shapes):
                raise Stage1FixtureError(f"gradient_matrix case {case_id} 参数集合错误")
            tensors = {name: _as_float_tensor(sample[name], field=f"{case_id}.samples[{sample_index}].{name}") for name in shapes}
            for name, tensor in tensors.items():
                if tuple(tensor.shape) != shapes[name]:
                    raise Stage1FixtureError(f"{case_id}.{name} shape 错误")
            parsed.append(TensorMap(tensors))
        weights = raw_case.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(parsed):
                raise Stage1FixtureError(f"gradient_matrix case {case_id} 权重数量错误")
            normalized_weights = tuple(float(item) for item in weights)
        else:
            normalized_weights = None
        cases[case_id] = {
            "samples": tuple(parsed),
            "weights": normalized_weights,
            "purpose": str(raw_case.get("purpose", "")),
        }
    return cases, shapes


def _analytic_evidence(manifest: Mapping[str, Any]) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    spec = manifest["analytic"]
    if not isinstance(spec, Mapping):
        raise Stage1FixtureError("analytic section 类型错误")
    initial = _nested_tensor_map(spec.get("initial"), field="analytic.initial")
    diagonal = _nested_tensor_map(spec.get("diagonal"), field="analytic.diagonal")
    linear = _nested_tensor_map(spec.get("linear"), field="analytic.linear")
    pre = _nested_tensor_map(spec.get("path_pre"), field="analytic.path_pre")
    post = _nested_tensor_map(spec.get("path_post"), field="analytic.path_post")
    initial.assert_compatible(diagonal)
    initial.assert_compatible(linear)
    initial.assert_compatible(pre)
    initial.assert_compatible(post)
    quadratic = QuadraticLossFixture(diagonal, linear, constant=float(spec.get("constant", 0.0)))
    leaves = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in initial.items()
    }
    point = TensorMap(leaves)
    analytic_gradient = quadratic.gradient_at(point)
    quadratic.loss(point).backward()
    autograd = TensorMap({name: leaves[name].grad.detach().clone() for name in leaves})
    steps = {
        name: _as_float_tensor(value, field=f"analytic.finite_difference_steps.{name}")
        for name, value in dict(spec.get("finite_difference_steps", {})).items()
    }
    finite = central_difference_gradient(quadratic.loss, point, step=steps)
    checks = [
        _check("analytic_autograd_fp64", _compare(autograd, analytic_gradient, natural_scale=16.0)),
        _check("analytic_finite_difference_fp64", _compare(finite, analytic_gradient, natural_scale=16.0)),
    ]

    constant_gradient = _nested_tensor_map(spec.get("constant_gradient"), field="analytic.constant_gradient")
    constant = ConstantGradientFixture(constant_gradient, constant=float(spec.get("constant", 0.0)))
    constant_path = PathSpec(pre, post, loss_id="stage1-s13-linear")
    constant_rules = (
        left_rule(),
        right_rule(),
        midpoint_rule(),
        trapezoid_rule(),
        simpson_rule(),
        gauss_legendre_rule(4),
    )
    constant_rule_results: dict[str, JSONValue] = {}
    expected_constant = constant.path_contribution(pre, post)
    for rule in constant_rules:
        result = integrate_path(
            constant_path,
            rule,
            lambda _alpha, state: constant.gradient_at(state),
            loss_fn=constant.loss,
        )
        comparison = _compare(result.signed, expected_constant, natural_scale=64.0)
        constant_rule_results[rule.name] = {
            "comparison": comparison,
            "completeness_absolute_residual": float(result.completeness_absolute_residual),
        }
        checks.append(_check(f"constant_path_{rule.name}", comparison))

    quadratic_path = PathSpec(pre, post, loss_id="stage1-s13-quadratic")
    quadratic_result = integrate_path(
        quadratic_path,
        trapezoid_rule(),
        lambda _alpha, state: quadratic.gradient_at(state),
        loss_fn=quadratic.loss,
    )
    expected_quadratic = quadratic.path_contribution(pre, post)
    quadratic_comparison = _compare(quadratic_result.signed, expected_quadratic, natural_scale=128.0)
    checks.append(_check("quadratic_path_coordinate_truth", quadratic_comparison))
    completeness = float(quadratic_result.completeness_absolute_residual)
    checks.append(_check("quadratic_path_completeness", completeness <= 1e-10, detail=f"residual={completeness:.3e}"))

    sgd_spec = spec.get("sgd")
    if not isinstance(sgd_spec, Mapping):
        raise Stage1FixtureError("analytic.sgd 缺失")
    learning_rate = float(sgd_spec["learning_rate"])
    weight_decay = float(sgd_spec["weight_decay"])
    if learning_rate <= 0 or weight_decay < 0:
        raise Stage1FixtureError("analytic.sgd 超参数非法")
    data_delta = {
        name: -learning_rate * analytic_gradient[name]
        for name in analytic_gradient
    }
    decay_delta = {
        name: -learning_rate * weight_decay * initial[name]
        for name in initial
    }
    total_delta = {
        name: data_delta[name] + decay_delta[name]
        for name in initial
    }
    updated = {
        name: initial[name] + total_delta[name]
        for name in initial
    }
    decomposition_max = max(
        float((total_delta[name] - data_delta[name] - decay_delta[name]).abs().max().item())
        for name in initial
    )
    no_decay_delta = {
        name: -learning_rate * analytic_gradient[name]
        for name in initial
    }
    checks.append(_check("known_sgd_no_decay_update", all(torch.equal(no_decay_delta[name], data_delta[name]) for name in initial)))
    checks.append(_check("known_sgd_decay_decomposition", decomposition_max <= 1e-15, detail=f"max_error={decomposition_max:.3e}"))
    return (
        {
            "schema_version": "stage1-analytic-fixture-v1",
            "initial": _tensor_map_to_wire(initial),
            "analytic_gradient": _tensor_map_to_wire(analytic_gradient),
            "autograd_gradient": _tensor_map_to_wire(autograd),
            "finite_difference_gradient": _tensor_map_to_wire(finite),
            "finite_difference_steps": {
                name: _tensor_to_wire(value) for name, value in steps.items()
            },
            "constant_path": {
                "expected": _tensor_map_to_wire(expected_constant),
                "rules": constant_rule_results,
            },
            "quadratic_path": {
                "expected": _tensor_map_to_wire(expected_quadratic),
                "actual": _tensor_map_to_wire(quadratic_result.signed),
                "comparison": quadratic_comparison,
                "completeness_absolute_residual": completeness,
            },
            "known_sgd": {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "data_delta": {name: _tensor_to_wire(value) for name, value in data_delta.items()},
                "weight_decay_delta": {name: _tensor_to_wire(value) for name, value in decay_delta.items()},
                "total_delta": {name: _tensor_to_wire(value) for name, value in total_delta.items()},
                "updated": {name: _tensor_to_wire(value) for name, value in updated.items()},
                "decomposition_max_error": decomposition_max,
            },
        },
        checks,
    )


def _tiny_evidence(manifest: Mapping[str, Any]) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    spec = manifest["tiny_transformer"]
    if not isinstance(spec, Mapping):
        raise Stage1FixtureError("tiny_transformer section 类型错误")
    model, registry, initial = build_tiny_transformer(spec)
    tokens, masks, sample_ids, groups = _tiny_inputs(spec)
    expected_state_hash = spec.get("initial_state_hash")
    actual_state_hash = _state_hash(initial)
    if expected_state_hash is not None and expected_state_hash != actual_state_hash:
        raise Stage1FixtureError("Tiny Transformer initial_state_hash 漂移")
    expected_registry_hash = spec.get("coordinate_registry_hash")
    if expected_registry_hash is not None and expected_registry_hash != registry.coordinate_registry_hash:
        raise Stage1FixtureError("Tiny Transformer coordinate_registry_hash 漂移")

    losses: dict[str, JSONValue] = {}
    sample_gradients: dict[str, TensorMap] = {}
    sample_counts: dict[str, int] = {}
    for index, sample_id in enumerate(sample_ids):
        _reset_model(model, initial)
        loss, count, gradient = _tiny_gradient(model, registry, tokens, masks, (index,))
        losses[sample_id] = {"loss": loss, "valid_target_tokens": count}
        sample_gradients[sample_id] = gradient
        sample_counts[sample_id] = count

    micro_gradients: list[TensorMap] = []
    micro_counts: list[int] = []
    micro_losses: list[float] = []
    for group in groups:
        _reset_model(model, initial)
        loss, count, gradient = _tiny_gradient(model, registry, tokens, masks, group)
        micro_losses.append(loss)
        micro_counts.append(count)
        micro_gradients.append(gradient)

    _reset_model(model, initial)
    full_loss, full_count, full_gradient = _tiny_gradient(model, registry, tokens, masks, tuple(range(len(sample_ids))))
    reconstructed_samples = {
        name: sum(
            sample_gradients[sample_id][name] * sample_counts[sample_id]
            for sample_id in sample_ids
        ) / float(sum(sample_counts.values()))
        for name in registry.eligible_names
    }
    reconstructed_micro = {
        name: sum(
            gradient[name] * count
            for gradient, count in zip(micro_gradients, micro_counts, strict=True)
        ) / float(sum(micro_counts))
        for name in registry.eligible_names
    }
    sample_reconstruction = TensorMap(reconstructed_samples, registry=registry)
    micro_reconstruction = TensorMap(reconstructed_micro, registry=registry)
    equal_micro = {
        name: sum(gradient[name] for gradient in micro_gradients) / float(len(micro_gradients))
        for name in registry.eligible_names
    }
    equal_micro_map = TensorMap(equal_micro, registry=registry)
    checks = [
        _check("tiny_sample_weighted_reconstruction", _compare(sample_reconstruction, full_gradient, natural_scale=64.0)),
        _check("tiny_microbatch_token_weighted_reconstruction", _compare(micro_reconstruction, full_gradient, natural_scale=64.0)),
    ]
    equal_comparison = _compare(equal_micro_map, full_gradient, natural_scale=64.0)
    unequal_padding_case = micro_counts[0] != micro_counts[1] if len(micro_counts) == 2 else False
    equal_is_distinct = not equal_comparison["passed"] if unequal_padding_case else True
    checks.append(_check("tiny_equal_microbatch_padding_counterexample", equal_is_distinct, detail=str(equal_comparison)))
    _reset_model(model, initial)
    after_hash = _state_hash({name: parameter.detach().clone() for name, parameter in model.named_parameters()})
    checks.append(_check("tiny_model_state_restored_after_oracle", after_hash == actual_state_hash))
    return (
        {
            "schema_version": "stage1-tiny-transformer-fixture-v1",
            "architecture": {
                "class": "TinyCausalTransformer",
                "vocab_size": int(spec["vocab_size"]),
                "hidden_size": int(spec["hidden_size"]),
                "dropout": 0.0,
                "random_layers": False,
            },
            "seed": int(spec["seed"]),
            "sample_ids": list(sample_ids),
            "tokens": tokens.tolist(),
            "target_masks": masks.tolist(),
            "microbatch_groups": [list(group) for group in groups],
            "valid_target_tokens": {sample_id: sample_counts[sample_id] for sample_id in sample_ids},
            "microbatch_valid_target_tokens": micro_counts,
            "losses": losses,
            "microbatch_losses": micro_losses,
            "full_batch_loss": full_loss,
            "full_batch_valid_target_tokens": full_count,
            "registry_manifest": registry.to_manifest(),
            "coordinate_registry_hash": registry.coordinate_registry_hash,
            "initial_state_hash": actual_state_hash,
            "sample_gradients": {
                sample_id: _tensor_map_to_wire(sample_gradients[sample_id])
                for sample_id in sample_ids
            },
            "microbatch_gradients": [_tensor_map_to_wire(value) for value in micro_gradients],
            "full_batch_gradient": _tensor_map_to_wire(full_gradient),
            "weighted_sample_reconstruction": _tensor_map_to_wire(sample_reconstruction),
            "weighted_microbatch_reconstruction": _tensor_map_to_wire(micro_reconstruction),
            "equal_microbatch_reconstruction": _tensor_map_to_wire(equal_micro_map),
            "equal_microbatch_comparison": equal_comparison,
        },
        checks,
    )


def _noise_evidence(manifest: Mapping[str, Any]) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    raw_specs = manifest["noise"]
    if not isinstance(raw_specs, list) or not raw_specs:
        raise Stage1FixtureError("noise fixture 列表为空")
    cases: list[dict[str, JSONValue]] = []
    checks: list[dict[str, JSONValue]] = []
    for raw in raw_specs:
        if not isinstance(raw, Mapping):
            raise Stage1FixtureError("noise fixture case 类型错误")
        fixture = ZeroMeanNoiseFixture(
            seed=int(raw["seed"]),
            sigma=float(raw["sigma"]),
            microbatch_size=int(raw["microbatch_size"]),
            microbatch_count=int(raw["microbatch_count"]),
            repetitions=int(raw["repetitions"]),
            coordinate_shapes={
                str(name): tuple(int(size) for size in shape)
                for name, shape in dict(raw["coordinate_shapes"]).items()
            },
        )
        raw_values: list[TensorMap] = []
        u_values: list[TensorMap] = []
        for samples in fixture.repetitions_iter():
            raw_map = _explicit_raw(samples)
            u_map = _explicit_pair_oracle(samples)
            raw_values.append(raw_map)
            u_values.append(u_map)
        coordinates = tuple(fixture.coordinate_shapes)
        raw_means = {
            name: torch.stack([item[name] for item in raw_values], dim=0).mean(dim=0)
            for name in coordinates
        }
        u_means = {
            name: torch.stack([item[name] for item in u_values], dim=0).mean(dim=0)
            for name in coordinates
        }
        raw_checks = {
            name: bool(
                torch.all(
                    (raw_means[name] - fixture.raw_expectation).abs()
                    <= 5.0 * fixture.raw_mean_standard_error
                )
                and torch.all(raw_means[name] > 0.0)
            )
            for name in coordinates
        }
        u_checks = {
            name: bool(torch.all(u_means[name].abs() <= 5.0 * fixture.u_mean_standard_error))
            for name in coordinates
        }
        checks.append(_check(f"noise_{raw['case_id']}_raw_theory", all(raw_checks.values()), detail=str(raw_checks)))
        checks.append(_check(f"noise_{raw['case_id']}_u_zero_mean", all(u_checks.values()), detail=str(u_checks)))
        cases.append(
            {
                "case_id": str(raw["case_id"]),
                "seed": fixture.seed,
                "sigma": fixture.sigma,
                "microbatch_size": fixture.microbatch_size,
                "microbatch_count": fixture.microbatch_count,
                "repetitions": fixture.repetitions,
                "raw_theoretical_mean": fixture.raw_expectation,
                "raw_mean_standard_error": fixture.raw_mean_standard_error,
                "u_theoretical_mean": 0.0,
                "u_mean_standard_error": fixture.u_mean_standard_error,
                "raw_means": _tensor_map_to_wire(TensorMap(raw_means)),
                "u_means": _tensor_map_to_wire(TensorMap(u_means)),
                "raw_checks": raw_checks,
                "u_checks": u_checks,
            }
        )
    return {"schema_version": "stage1-zero-mean-noise-v1", "cases": cases}, checks


def _gradient_matrix_evidence(manifest: Mapping[str, Any]) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    cases, shapes = _gradient_cases(manifest)
    results: dict[str, JSONValue] = {}
    checks: list[dict[str, JSONValue]] = []
    for case_id, case in cases.items():
        samples = case["samples"]
        weights = case["weights"]
        mean = _explicit_mean(samples)
        record: dict[str, JSONValue] = {
            "sample_count": len(samples),
            "purpose": case["purpose"],
            "mean_gradient": _tensor_map_to_wire(mean),
            "samples": [_tensor_map_to_wire(sample) for sample in samples],
        }
        if len(samples) >= 2:
            pair_oracle = _explicit_pair_oracle(samples)
            raw_oracle = _explicit_raw(samples)
            raw_comparison = _compare(raw_oracle, mean.map(torch.square), natural_scale=1e10)
            u_comparison = _compare(pair_oracle, _explicit_pair_oracle(samples), natural_scale=1e10)
            record.update(
                {
                    "raw_oracle": _tensor_map_to_wire(raw_oracle),
                    "u_oracle": _tensor_map_to_wire(pair_oracle),
                    "raw_identity_comparison": raw_comparison,
                    "u_identity_comparison": u_comparison,
                }
            )
            checks.append(_check(f"gradient_{case_id}_raw", raw_comparison))
            checks.append(_check(f"gradient_{case_id}_u", u_comparison))
            if weights is not None:
                weighted_oracle = _explicit_pair_oracle(samples, weights)
                weighted_comparison = _compare(
                    weighted_oracle,
                    _explicit_pair_oracle(samples, weights),
                    natural_scale=1e10,
                )
                record["weights"] = list(weights)
                record["weighted_u_oracle"] = _tensor_map_to_wire(weighted_oracle)
                record["weighted_u_identity_comparison"] = weighted_comparison
                checks.append(_check(f"gradient_{case_id}_weighted_u", weighted_comparison))
        else:
            try:
                _explicit_pair_oracle(samples)
            except Stage1FixtureError as error:
                record["m1_rejection"] = {"status": "PASS", "error_type": type(error).__name__, "message": str(error)}
                checks.append(_check(f"gradient_{case_id}_m1_rejected", True))
            else:
                record["m1_rejection"] = {"status": "FAIL"}
                checks.append(_check(f"gradient_{case_id}_m1_rejected", False))
        results[case_id] = record

    negative_case = results.get("negative-u-m2")
    negative_values: list[float] = []
    if isinstance(negative_case, Mapping):
        oracle = negative_case.get("u_oracle")
        if isinstance(oracle, Mapping):
            for item in oracle.values():
                if isinstance(item, Mapping):
                    values = item.get("values")
                    if isinstance(values, list):
                        negative_values.extend(float(value) for value in values)
    checks.append(_check("gradient_negative_u_is_retained", any(value < 0 for value in negative_values)))

    permutation_source = cases.get("varied-m4")
    if permutation_source is not None:
        baseline = _explicit_pair_oracle(permutation_source["samples"])
        permutation_results: list[dict[str, JSONValue]] = []
        for order in ((3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 2, 0)):
            permuted = tuple(permutation_source["samples"][index] for index in order)
            comparison = _compare(_explicit_pair_oracle(permuted), baseline, natural_scale=1e10)
            permutation_results.append({"order": list(order), "comparison": comparison})
        results["permutations"] = permutation_results
        checks.append(_check("gradient_order_permutation_invariance", all(item["comparison"]["passed"] for item in permutation_results)))
    return (
        {
            "schema_version": "stage1-gradient-matrix-fixture-v1",
            "coordinate_shapes": {name: list(shape) for name, shape in shapes.items()},
            "cases": results,
        },
        checks,
    )


def _bundle_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("bundle_hash", None)
    return canonical_json_hash(body)


def _validation_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("report_hash", None)
    return canonical_json_hash(body)


def build_stage1_s13_evidence(
    source_root: str | Path,
    *,
    producer_commit: str = "0" * 40,
    scope: str = "local_fixture",
    upstream_evidence: Mapping[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    """Build the three role-specific S1.3 evidence payloads."""

    root = Path(source_root).resolve()
    manifest = _load_manifest(root)
    matrix, matrix_checks = _gradient_matrix_evidence(manifest)
    analytic, analytic_checks = _analytic_evidence(manifest)
    tiny, tiny_checks = _tiny_evidence(manifest)
    noise, noise_checks = _noise_evidence(manifest)
    all_checks = [*matrix_checks, *analytic_checks, *tiny_checks, *noise_checks]
    if not all(item["status"] == "PASS" for item in all_checks):
        failed = [str(item["check_id"]) for item in all_checks if item["status"] != "PASS"]
        raise Stage1FixtureError("S1.3 oracle validation failed: " + ",".join(failed))

    # The committed manifest is immutable: execution metadata belongs in the
    # validation report, otherwise the manifest_hash would no longer describe
    # the payload that is being published as the fixture contract.
    fixture_manifest = dict(manifest)

    oracle_bundle: dict[str, JSONValue] = {
        "schema_version": ORACLE_SCHEMA,
        "fixture_set_id": FIXTURE_SET_ID,
        "producer_commit": producer_commit,
        "scope": scope,
        "fixture_manifest_hash": str(manifest["manifest_hash"]),
        "gradient_matrix": matrix,
        "analytic": analytic,
        "tiny_transformer": tiny,
        "zero_mean_noise": noise,
        "offline_recompute": {
            "algorithm": "explicit_fp64_loops_over_serialized_inputs",
            "formal_estimator_imported": False,
            "recomputed_from_saved_values": True,
        },
    }
    oracle_bundle["bundle_hash"] = _bundle_hash(oracle_bundle)
    validation: dict[str, JSONValue] = {
        "schema_version": VALIDATION_SCHEMA,
        "gate_id": "G1-ORACLE",
        "status": "PASS",
        "scope": scope,
        "formal_eligible": scope == "formal",
        "producer_commit": producer_commit,
        "fixture_manifest_hash": str(manifest["manifest_hash"]),
        "oracle_bundle_hash": str(oracle_bundle["bundle_hash"]),
        "upstream_evidence": dict(upstream_evidence or {}),
        "source_file_hashes": _git_source_hashes(root),
        "checks": all_checks,
        "check_count": len(all_checks),
        "passed_check_count": sum(item["status"] == "PASS" for item in all_checks),
        "pythia_14m": {
            "status": "DEFERRED_TO_S1.7",
            "model_revision": PYTHIA_STEP0_REVISION,
            "consumed_by_gate": False,
        },
        "replay_contract": {
            "saved_bundle_can_be_recomputed_without_sampling": True,
            "saved_bundle_can_be_recomputed_without_training_loop": True,
        },
    }
    validation["report_hash"] = _validation_hash(validation)
    return {
        "fixture_manifest": fixture_manifest,
        "oracle_bundle": oracle_bundle,
        "oracle_validation_report": validation,
    }


def recompute_serialized_oracle_bundle(bundle: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Recompute core matrix/U values from an already saved bundle.

    This intentionally uses only explicit Python loops and the serialised gradient
    arrays.  It is used by the offline replay test and by the formal publisher.
    """

    if bundle.get("schema_version") != ORACLE_SCHEMA:
        raise Stage1FixtureError("oracle bundle schema_version 错误")
    expected_hash = bundle.get("bundle_hash")
    if not isinstance(expected_hash, str) or expected_hash != _bundle_hash(bundle):
        raise Stage1FixtureError("oracle bundle hash 不匹配")
    matrix = bundle.get("gradient_matrix")
    if not isinstance(matrix, Mapping):
        raise Stage1FixtureError("oracle bundle 缺少 gradient_matrix")
    replay: dict[str, JSONValue] = {}
    cases = matrix.get("cases")
    if not isinstance(cases, Mapping):
        raise Stage1FixtureError("oracle bundle gradient cases 类型错误")
    for case_id, raw_case in cases.items():
        if case_id == "permutations" or not isinstance(raw_case, Mapping):
            continue
        raw_samples = raw_case.get("samples")
        if not isinstance(raw_samples, list):
            raise Stage1FixtureError(f"oracle bundle case {case_id} samples 缺失")
        samples = tuple(
            _tensor_map_from_wire(sample, field=f"bundle.gradient_matrix.cases.{case_id}.samples[{index}]")
            for index, sample in enumerate(raw_samples)
        )
        entry: dict[str, JSONValue] = {"mean_gradient": _tensor_map_to_wire(_explicit_mean(samples))}
        if len(samples) >= 2:
            entry["u_oracle"] = _tensor_map_to_wire(_explicit_pair_oracle(samples))
            weights = raw_case.get("weights")
            if isinstance(weights, list):
                entry["weighted_u_oracle"] = _tensor_map_to_wire(
                    _explicit_pair_oracle(samples, [float(item) for item in weights])
                )
        replay[str(case_id)] = entry
    return {
        "schema_version": "stage1-oracle-replay-v1",
        "fixture_set_id": bundle.get("fixture_set_id"),
        "source_bundle_hash": expected_hash,
        "gradient_matrix": replay,
    }


def validate_stage1_s13_evidence(evidence: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Validate the role-specific payloads after a canonical save/load roundtrip."""

    manifest = evidence.get("fixture_manifest")
    bundle = evidence.get("oracle_bundle")
    report = evidence.get("oracle_validation_report")
    if not isinstance(manifest, Mapping) or not isinstance(bundle, Mapping) or not isinstance(report, Mapping):
        raise Stage1FixtureError("S1.3 evidence 三个 payload 不完整")
    if manifest.get("schema_version") != FIXTURE_SCHEMA:
        raise Stage1FixtureError("S1.3 evidence manifest schema 错误")
    manifest_body = dict(manifest)
    manifest_hash = manifest_body.pop("manifest_hash", None)
    if not isinstance(manifest_hash, str) or manifest_hash != canonical_json_hash(manifest_body):
        raise Stage1FixtureError("S1.3 evidence manifest hash 不匹配")
    if bundle.get("fixture_manifest_hash") != manifest.get("manifest_hash"):
        raise Stage1FixtureError("S1.3 bundle 未绑定 manifest hash")
    if report.get("oracle_bundle_hash") != bundle.get("bundle_hash"):
        raise Stage1FixtureError("S1.3 report 未绑定 bundle hash")
    report_hash = report.get("report_hash")
    if not isinstance(report_hash, str) or report_hash != _validation_hash(report):
        raise Stage1FixtureError("S1.3 validation report hash 不匹配")
    checks = report.get("checks")
    if (
        report.get("status") != "PASS"
        or not isinstance(checks, list)
        or report.get("check_count") != len(checks)
        or report.get("passed_check_count") != len(checks)
        or any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in checks)
    ):
        raise Stage1FixtureError("S1.3 validation report 未通过全部 checks")
    replay = recompute_serialized_oracle_bundle(bundle)
    return {
        "schema_version": "stage1-oracle-replay-validation-v1",
        "manifest_hash": manifest.get("manifest_hash"),
        "bundle_hash": bundle.get("bundle_hash"),
        "report_hash": report.get("report_hash"),
        "replay_hash": canonical_json_hash(replay),
        "replay": replay,
        "report_status": report.get("status"),
    }


__all__ = [
    "FIXTURE_SCHEMA",
    "FIXTURE_SET_ID",
    "ORACLE_SCHEMA",
    "PYTHIA_STEP0_REVISION",
    "Stage1FixtureError",
    "TinyCausalTransformer",
    "VALIDATION_SCHEMA",
    "build_stage1_s13_evidence",
    "build_tiny_transformer",
    "recompute_serialized_oracle_bundle",
    "validate_stage1_s13_evidence",
]
