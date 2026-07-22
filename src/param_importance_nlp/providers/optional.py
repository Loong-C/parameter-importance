"""外部 ML 生态的延迟依赖边界。

本机 CPU core profile 不安装 Transformers、Datasets、Accelerate、Safetensors
或 TensorBoard。导入 ``param_importance_nlp`` 与 ``providers`` 因而不能触发这些
包；只有显式构造相应 adapter 时才解析模块，并以结构化
``DependencyUnavailable`` 失败。该边界也禁止 adapter 静默联网下载资产。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType

from ..contracts import DependencyUnavailable


def require_optional_dependency(
    module_name: str,
    *,
    feature: str,
    install_extra: str | None = "server",
) -> ModuleType:
    """延迟导入一个可选模块，并保留稳定机器错误码。"""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DependencyUnavailable(
            module_name, feature=feature, install_extra=install_extra
        ) from exc


@dataclass(frozen=True, slots=True)
class HuggingFaceDependencies:
    """已显式加载的 HF 模块集合；构造本身不访问网络或资产。"""

    transformers: ModuleType
    datasets: ModuleType
    accelerate: ModuleType


def load_huggingface_dependencies() -> HuggingFaceDependencies:
    """为未来真实 provider 加载依赖，不调用 ``from_pretrained``。"""

    return HuggingFaceDependencies(
        transformers=require_optional_dependency(
            "transformers", feature="huggingface_gradient_provider"
        ),
        datasets=require_optional_dependency(
            "datasets", feature="huggingface_gradient_provider"
        ),
        accelerate=require_optional_dependency(
            "accelerate", feature="huggingface_gradient_provider"
        ),
    )


def load_safetensors_module() -> ModuleType:
    """加载可选 tensor codec；默认安全 bundle 不依赖它。"""

    return require_optional_dependency("safetensors", feature="safetensors_codec")


def load_tensorboard_module() -> ModuleType:
    """加载可选日志 adapter；JSONL 机器真值不依赖它。"""

    return require_optional_dependency("tensorboard", feature="tensorboard_logging")


__all__ = [
    "HuggingFaceDependencies",
    "load_huggingface_dependencies",
    "load_safetensors_module",
    "load_tensorboard_module",
    "require_optional_dependency",
]
