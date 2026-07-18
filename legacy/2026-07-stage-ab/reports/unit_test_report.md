# Stage A unit-test report

- Status: **PASS**
- Git commit: `78329802193d89e121a2e275d8d3ce312bbe1255`
- Generated UTC: `2026-07-17T04:25:03Z`
- Python: `3.12.3`
- PyTorch: `2.12.1+cu126`; CUDA: `12.6`
- Command: `/home/sophgo13/cjl/storage/parameter-importance/envs/parameter-importance/bin/python -m pytest -q --ignore=tests/test_pythia_provider_integration.py --junitxml=/tmp/stage-a-tests-4pajzqbe/pytest.xml`
- Elapsed seconds: `42.527`
- Tests: `156`; failures: `0`; errors: `0`; skipped: `0`

A pass requires a zero pytest exit code, at least one collected test, and zero failures, errors, or skips. The real Pythia provider contract is deliberately excluded here because it is executed against the completed Stage-A trajectory as a separate measured gate.
