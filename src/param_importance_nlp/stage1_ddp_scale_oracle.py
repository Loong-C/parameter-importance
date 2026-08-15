"""Independent pre-route gradient-scale oracle for the S1.8 fixture.

This is intentionally not a DDP worker and imports neither ``stage1_ddp`` nor
its sufficient-statistic implementation.  It observes the maximum absolute
unit-gradient scale on the frozen S1.7 records before the A/B/C/D route plans
and fixture are created.  The resulting hash and value bind every later
natural scale; no value is inferred from a route being validated.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch

from .atomic import sha256_file
from .contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json


TASK_ID = "stage1.08_ddp_and_gradient_accumulation"
PLAN_SCHEMA = "stage1-s1-8-pre-route-scale-plan-v2"
REPORT_SCHEMA = "stage1-s1-8-pre-route-gradient-scale-oracle-v2"
MICROBATCH_COUNT = 8
OPTIMIZER_CONDITIONING = {
    "schema_version": "stage1-s1-8-optimizer-conditioning-v2",
    "precision_profile": "T32_DISTRIBUTED",
    "optimizer_type": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1.0e-4,
    "foreach": False,
    "fused": False,
    "execution_order": "single_tensor_decoupled_decay_then_adamw_addcdiv_fp32",
    "selection_method": "r16_route_specific_mean_epsilon_sensitivity",
    "diagnostic_attempt_id": "formal-20260815-s18-v1-r16",
    "failed_attempt_artifact_hash": "81674ea39a2b67fcfa4a0294e5d59f75882908cb3f52384a8ae5c85bc7960789",
    "failed_attempt_phase": "rank_gradient_checksum_cardinality",
    "r16_optimizer_epsilon": 1.0e-8,
    "sensitivity_method": "read_only_persisted_route_mean_simulation",
    "sensitivity_t32_peer_violation_counts": {
        "1e-8": {"equal": 8, "weighted": 9},
        "1e-6": {"equal": 6, "weighted": 6},
        "1e-5": {"equal": 2, "weighted": 0},
        "1e-4": {"equal": 0, "weighted": 0},
    },
    "selected_epsilon_first_t32_zero_violation": 1.0e-4,
}


class Stage1S18ScaleOracleError(RuntimeError):
    """Raised when the pre-route scale producer cannot prove its inputs."""


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1S18ScaleOracleError(f"S18_SCALE_{field.upper()}_OBJECT_REQUIRED")
    return dict(value)


def _exact_contract_value(value: object, expected: object) -> bool:
    """Reject jointly rehashed bool/int and nested configuration drift."""

    if isinstance(expected, Mapping):
        return isinstance(value, Mapping) and set(value) == set(expected) and all(
            _exact_contract_value(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            _exact_contract_value(item, reference) for item, reference in zip(value, expected, strict=True)
        )
    return type(value) is type(expected) and value == expected


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise Stage1S18ScaleOracleError(f"S18_SCALE_{field.upper()}_HASH_INVALID")
    return value


def _with_hash(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    if "artifact_hash" in body:
        raise Stage1S18ScaleOracleError("S18_SCALE_ARTIFACT_HASH_DERIVED")
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def validate_plan(value: Mapping[str, object], *, path: Path) -> dict[str, Any]:
    plan = _mapping(value, field="plan")
    expected = {
        "schema_version", "task_id", "execution_commit", "fixture_tokens_ref", "fixture_tokens_sha256",
        "selected_token_sha256", "upstream_token_sha256", "cases", "optimizer_conditioning", "model_root", "output_file", "run_token", "visible_gpu_uuids", "artifact_hash",
    }
    declared = plan.pop("artifact_hash", None)
    if set(plan) != expected - {"artifact_hash"} or plan.get("schema_version") != PLAN_SCHEMA or plan.get("task_id") != TASK_ID or declared != canonical_json_hash(plan):
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_SCHEMA_INVALID")
    plan["artifact_hash"] = declared
    if not _exact_contract_value(_mapping(plan.get("optimizer_conditioning"), field="plan.optimizer_conditioning"), OPTIMIZER_CONDITIONING):
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_OPTIMIZER_CONDITIONING_INVALID")
    if not isinstance(plan.get("execution_commit"), str) or len(str(plan["execution_commit"])) != 40:
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_COMMIT_INVALID")
    _hash(plan.get("fixture_tokens_sha256"), field="fixture_tokens")
    if not isinstance(plan.get("fixture_tokens_ref"), str) or Path(str(plan["fixture_tokens_ref"])).is_absolute():
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_TOKEN_REFERENCE_INVALID")
    tokens = (path.parent / str(plan["fixture_tokens_ref"])).resolve()
    if not tokens.is_file() or sha256_file(tokens) != plan["fixture_tokens_sha256"]:
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_TOKEN_FILE_HASH_INVALID")
    upstream_hashes = _mapping(plan.get("upstream_token_sha256"), field="plan.upstream_token_sha256")
    token_hashes = _mapping(plan.get("selected_token_sha256"), field="plan.selected_token_sha256")
    if set(upstream_hashes) != {str(index) for index in range(16)} or set(token_hashes) != {str(index) for index in range(MICROBATCH_COUNT)}:
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_TOKEN_HASH_SET_INVALID")
    if any(token_hashes[str(index)] != upstream_hashes[str(index)] for index in range(MICROBATCH_COUNT)):
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_SELECTED_TOKEN_BINDING_INVALID")
    for value_hash in (*token_hashes.values(), *upstream_hashes.values()):
        _hash(value_hash, field="token")
    cases = _mapping(plan.get("cases"), field="plan.cases")
    if set(cases) != {"equal", "weighted"}:
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_CASE_SET_INVALID")
    for case in ("equal", "weighted"):
        item = _mapping(cases[case], field=f"plan.cases.{case}")
        suffixes = item.get("label_ignore_suffixes")
        if not isinstance(suffixes, list) or len(suffixes) != MICROBATCH_COUNT or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 2048 for value in suffixes):
            raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_SUFFIXES_INVALID")
    if not isinstance(plan.get("model_root"), str) or not str(plan["model_root"]):
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_MODEL_ROOT_INVALID")
    if not isinstance(plan.get("output_file"), str) or Path(str(plan["output_file"])).is_absolute() or Path(str(plan["output_file"])).name != str(plan["output_file"]):
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_OUTPUT_INVALID")
    if not isinstance(plan.get("run_token"), str) or len(str(plan["run_token"])) != 64:
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_RUN_TOKEN_INVALID")
    visible = plan.get("visible_gpu_uuids")
    if not isinstance(visible, list) or len(visible) != 1 or not isinstance(visible[0], str) or not visible[0].startswith("GPU-"):
        raise Stage1S18ScaleOracleError("S18_SCALE_PLAN_UUID_INVALID")
    return plan


def _load_tokens(path: Path, hashes: Mapping[str, str]) -> dict[int, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - server-only dependency
        raise Stage1S18ScaleOracleError("S18_SCALE_SAFETENSORS_UNAVAILABLE") from error
    values = load_file(str(path), device="cpu")
    required = {f"record_{index:012d}" for index in range(16)}
    if set(values) != required:
        raise Stage1S18ScaleOracleError("S18_SCALE_TENSOR_SET_INVALID")
    result: dict[int, torch.Tensor] = {}
    for index in range(MICROBATCH_COUNT):
        value = values[f"record_{index:012d}"]
        digest = hashlib.sha256(value.contiguous().numpy().tobytes(order="C")).hexdigest()
        if value.dtype != torch.int64 or tuple(value.shape) != (2049,) or digest != hashes[str(index)]:
            raise Stage1S18ScaleOracleError(f"S18_SCALE_TOKEN_INVALID:{index}")
        result[index] = value.contiguous()
    return result


def _load_model(model_root: str, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - server-only dependency
        raise Stage1S18ScaleOracleError("S18_SCALE_TRANSFORMERS_UNAVAILABLE") from error
    model = AutoModelForCausalLM.from_pretrained(model_root, local_files_only=True, torch_dtype=torch.float32).to(device)
    for name in ("attention_dropout", "hidden_dropout"):
        if float(getattr(model.config, name, math.nan)) != 0.0:
            raise Stage1S18ScaleOracleError(f"S18_SCALE_DROPOUT_ACTIVE:{name}")
    model.train()
    return model


def _tensor_map_digest(values: Mapping[str, torch.Tensor]) -> str:
    """Match the route worker's name/dtype/shape/data parameter digest."""

    digest = hashlib.sha256(b"stage1-s1-8-tensor-map-v1\0")
    for name in sorted(values):
        value = values[name].detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _stable_l2(values: Mapping[str, torch.Tensor]) -> float:
    scale, ssq = 0.0, 1.0
    for value in values.values():
        flat = value.detach().to(torch.float64).reshape(-1)
        block_scale = float(flat.abs().max().item()) if flat.numel() else 0.0
        if block_scale == 0.0:
            continue
        block_ssq = float((flat / block_scale).square().sum().item())
        if scale == 0.0:
            scale, ssq = block_scale, block_ssq
        elif scale < block_scale:
            ssq = block_ssq + ssq * (scale / block_scale) ** 2; scale = block_scale
        else:
            ssq += block_ssq * (block_scale / scale) ** 2
    return 0.0 if scale == 0.0 else scale * math.sqrt(ssq)


def execute_scale_oracle(plan_path: str | Path) -> dict[str, object]:
    """Run the independent single-GPU autograd scale observation once."""

    plan_path = Path(plan_path).resolve(strict=True)
    plan = validate_plan(_mapping(load_canonical_json(plan_path), field="plan"), path=plan_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["visible_gpu_uuids"][0]:
        raise Stage1S18ScaleOracleError("S18_SCALE_CUDA_VISIBLE_DEVICES_DRIFT")
    if not torch.cuda.is_available():
        raise Stage1S18ScaleOracleError("S18_SCALE_CUDA_REQUIRED")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokens = _load_tokens(plan_path.parent / str(plan["fixture_tokens_ref"]), _mapping(plan["selected_token_sha256"], field="plan.selected_token_sha256"))
    torch.manual_seed(1707)
    model = _load_model(str(plan["model_root"]), device)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not parameters:
        raise Stage1S18ScaleOracleError("S18_SCALE_ELIGIBLE_PARAMETER_SET_EMPTY")
    registry_hash = canonical_json_hash({
        "names": [name for name, _ in parameters],
        "shapes": {name: list(parameter.shape) for name, parameter in parameters},
        "dtypes": {name: str(parameter.dtype) for name, parameter in parameters},
    })
    optimizer_conditioning = _mapping(plan["optimizer_conditioning"], field="plan.optimizer_conditioning")
    learning_rate = float(optimizer_conditioning["learning_rate"])
    weight_decay = float(optimizer_conditioning["weight_decay"])
    betas = tuple(float(value) for value in optimizer_conditioning["betas"])
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in parameters], lr=learning_rate, weight_decay=weight_decay,
        betas=betas, eps=float(optimizer_conditioning["eps"]),
        foreach=bool(optimizer_conditioning["foreach"]), fused=bool(optimizer_conditioning["fused"]),
    )
    records: list[dict[str, object]] = []
    maximum, maximum_case, maximum_microbatch, maximum_parameter = 0.0, "", -1, ""
    maximum_data_update, maximum_data_case, maximum_data_parameter = 0.0, "", ""
    case_pre_parameter_checksums: dict[str, str] = {}
    case_post_parameter_checksums: dict[str, str] = {}
    for case in ("equal", "weighted"):
        suffixes = _mapping(_mapping(plan["cases"], field="plan.cases")[case], field=f"plan.cases.{case}")["label_ignore_suffixes"]
        weights = [2048 - int(value) for value in suffixes]
        case_pre_parameter_checksums[case] = _tensor_map_digest({name: parameter for name, parameter in parameters})
        accumulated = {name: torch.zeros_like(parameter) for name, parameter in parameters}
        for index in range(MICROBATCH_COUNT):
            sequence = tokens[index].unsqueeze(0).to(device, non_blocking=True)
            inputs, labels = sequence[:, :-1], sequence[:, 1:].clone()
            suffix = int(suffixes[index])
            if suffix:
                labels[:, -suffix:] = -100
            output = model(input_ids=inputs, labels=labels)
            if output.loss is None:
                raise Stage1S18ScaleOracleError("S18_SCALE_MODEL_LOSS_MISSING")
            gradients = torch.autograd.grad(output.loss, [parameter for _, parameter in parameters], allow_unused=False)
            unit_max, unit_parameter = 0.0, ""
            for (name, _), gradient in zip(parameters, gradients, strict=True):
                if not bool(torch.isfinite(gradient).all()):
                    raise Stage1S18ScaleOracleError(f"S18_SCALE_GRADIENT_NONFINITE:{case}:{index}:{name}")
                value = float(gradient.detach().abs().max().item())
                if value > unit_max:
                    unit_max, unit_parameter = value, name
                accumulated[name].add_(gradient, alpha=float(weights[index]))
            records.append({"case": case, "microbatch_id": index, "maximum_abs_gradient": unit_max, "parameter": unit_parameter})
            if unit_max > maximum:
                maximum, maximum_case, maximum_microbatch, maximum_parameter = unit_max, case, index, unit_parameter
        denominator = sum(weights)
        mean = {name: value / denominator for name, value in accumulated.items()}
        norm = _stable_l2(mean)
        clip_factor = min(1.0, 1.0 / (norm + 1.0e-6))
        if not 0.0 < clip_factor <= 1.0:
            raise Stage1S18ScaleOracleError("S18_SCALE_CLIP_FACTOR_INVALID")
        optimizer.zero_grad(set_to_none=True)
        pre_parameters = {name: parameter.detach().clone() for name, parameter in parameters}
        for name, parameter in parameters:
            parameter.grad = (mean[name] * clip_factor).to(dtype=parameter.dtype).clone()
        optimizer.step()
        for name, parameter in parameters:
            # AdamW's decoupled decay is exactly -eta*lambda*theta_pre; remove
            # it to freeze the data-update design scale independently of U.
            data_update = (parameter.detach() - pre_parameters[name]) + learning_rate * weight_decay * pre_parameters[name]
            value = float(data_update.abs().max().item())
            if value > maximum_data_update:
                maximum_data_update, maximum_data_case, maximum_data_parameter = value, case, name
        case_post_parameter_checksums[case] = _tensor_map_digest({name: parameter for name, parameter in parameters})
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise Stage1S18ScaleOracleError("S18_SCALE_MAXIMUM_NOT_POSITIVE")
    if not math.isfinite(maximum_data_update) or maximum_data_update <= 0.0:
        raise Stage1S18ScaleOracleError("S18_SCALE_DATA_UPDATE_MAXIMUM_NOT_POSITIVE")
    result = _with_hash({
        "schema_version": REPORT_SCHEMA, "status": "PASS", "task_id": TASK_ID,
        "execution_commit": plan["execution_commit"], "run_token": plan["run_token"],
        "fixture_tokens_sha256": plan["fixture_tokens_sha256"], "selected_token_sha256": plan["selected_token_sha256"], "upstream_token_sha256": plan["upstream_token_sha256"],
        "visible_gpu_uuid": plan["visible_gpu_uuids"][0], "method": "independent_pre_route_single_gpu_autograd_oracle", "optimizer_conditioning": plan["optimizer_conditioning"],
        "unit_count": len(records), "unit_records": records, "maximum_unit_gradient_abs": maximum,
        "maximum_case": maximum_case, "maximum_microbatch_id": maximum_microbatch, "maximum_parameter": maximum_parameter,
        "maximum_abs_data_update": maximum_data_update, "maximum_data_update_case": maximum_data_case,
        "maximum_data_update_parameter": maximum_data_parameter,
        "parameter_registry_hash": registry_hash, "case_pre_parameter_checksums": case_pre_parameter_checksums,
        "case_post_parameter_checksums": case_post_parameter_checksums,
    })
    output = plan_path.parent / str(plan["output_file"])
    if output.exists():
        raise Stage1S18ScaleOracleError("S18_SCALE_OUTPUT_COLLISION")
    write_canonical_json(output, result)
    return result


__all__ = ["OPTIMIZER_CONDITIONING", "PLAN_SCHEMA", "REPORT_SCHEMA", "Stage1S18ScaleOracleError", "TASK_ID", "execute_scale_oracle", "validate_plan"]
