"""梯度数据源协议与本机合成实现。

本分区描述固定状态梯度源以及训练所需的模型/数据协议。真实 Hugging Face
适配器只接受本地目录；导入本包本身永远不会触发 Transformers、Datasets
或网络访问。
"""

from .protocols import (
    FixedStateGradientProvider,
    FrozenSampleResolver,
    GradientBatch,
    GradientProvider,
)
from .optional import (
    HuggingFaceDependencies,
    load_huggingface_dependencies,
    load_safetensors_module,
    load_tensorboard_module,
    require_optional_dependency,
)
from .synthetic import SyntheticGradientProvider
from .training import (
    BatchCursor,
    CausalLMEvaluator,
    ClassificationEvaluator,
    DatasetAdapter,
    DeterministicBatchCursor,
    InMemoryDatasetAdapter,
    ModelAdapter,
    OfflineHuggingFaceModelAdapter,
    OfflineTokenizer,
    PrefetchBatchCursor,
    PretokenizedJsonlDatasetAdapter,
    TaskEvaluator,
    TorchModelAdapter,
    TrainingMicrobatch,
    configure_batch_cursor,
)
from .tiny import (
    TinyCausalLM,
    TinySequenceClassifier,
    TinyTrainingFixture,
    build_tiny_training_fixture,
)
from .fixed_state_torch import (
    InMemoryFrozenSampleResolver,
    TorchFixedStateGradientProvider,
)
from .huggingface_offline import (
    HuggingFaceTaskMetricEvaluator,
    PretokenizedGlueDatasetAdapter,
    PretokenizedPileDatasetAdapter,
    hash_local_directory,
)

__all__ = [
    "BatchCursor",
    "CausalLMEvaluator",
    "ClassificationEvaluator",
    "DatasetAdapter",
    "DeterministicBatchCursor",
    "FixedStateGradientProvider",
    "FrozenSampleResolver",
    "GradientBatch",
    "GradientProvider",
    "HuggingFaceDependencies",
    "HuggingFaceTaskMetricEvaluator",
    "InMemoryFrozenSampleResolver",
    "InMemoryDatasetAdapter",
    "ModelAdapter",
    "OfflineHuggingFaceModelAdapter",
    "OfflineTokenizer",
    "PrefetchBatchCursor",
    "PretokenizedJsonlDatasetAdapter",
    "PretokenizedGlueDatasetAdapter",
    "PretokenizedPileDatasetAdapter",
    "SyntheticGradientProvider",
    "TaskEvaluator",
    "TinyCausalLM",
    "TinySequenceClassifier",
    "TinyTrainingFixture",
    "TorchModelAdapter",
    "TorchFixedStateGradientProvider",
    "TrainingMicrobatch",
    "build_tiny_training_fixture",
    "configure_batch_cursor",
    "hash_local_directory",
    "load_huggingface_dependencies",
    "load_safetensors_module",
    "load_tensorboard_module",
    "require_optional_dependency",
]
