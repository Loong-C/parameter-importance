from __future__ import annotations

import pytest

from param_importance_nlp.contracts import DependencyUnavailable
from param_importance_nlp.providers import require_optional_dependency


def test_optional_dependency_is_lazy_and_structured() -> None:
    with pytest.raises(DependencyUnavailable) as captured:
        require_optional_dependency(
            "package_that_does_not_exist_for_fixture",
            feature="fixture_adapter",
            install_extra="server",
        )
    error = captured.value
    assert error.dependency == "package_that_does_not_exist_for_fixture"
    assert error.feature == "fixture_adapter"
    assert str(error).startswith("DEPENDENCY_UNAVAILABLE:")
