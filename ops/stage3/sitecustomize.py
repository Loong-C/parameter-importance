"""Auditable opt-in startup hook for the Stage 3 verification cache.

This file is intentionally tracked beside the cache control module.  Python
loads ``sitecustomize`` before application imports when the repository's
``ops`` package is on ``PYTHONPATH``; the installer emits a self-contained
copy for deployments that expose only the source tree.  With no explicit
cache-root environment variable the installer is a no-op.
"""

try:
    from ops.stage3.file_verification_cache import install_file_verification_cache
except ImportError:
    # When this directory itself is placed on ``PYTHONPATH`` (the usual
    # sitecustomize deployment shape), the repository package prefix is not
    # importable yet.  The sibling module remains the same tracked source.
    from file_verification_cache import install_file_verification_cache


try:
    install_file_verification_cache()
except Exception:
    # Optional startup acceleration must never change normal interpreter
    # startup or Pythia's authoritative digest checks.
    pass
