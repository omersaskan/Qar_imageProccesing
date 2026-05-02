"""Sprint 4.6 — failure classification helper tests."""
from __future__ import annotations

import pytest

from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.failures import (
    InsufficientReconstructionError,
    MissingArtifactError,
    RuntimeReconstructionError,
    TexturingFailed,
)


classify = ReconstructionRunner._classify_attempt_failure


def test_classify_native_crash_by_exit_code():
    exc = RuntimeReconstructionError("OpenMVS exit code 3221226505 during TextureMesh")
    cls, code, summary = classify(exc)
    assert cls == "native_crash"
    assert code == 3221226505
    assert "3221226505" in summary


def test_classify_native_crash_by_hex_signature():
    exc = RuntimeReconstructionError("Process terminated with 0xC0000005 (access violation)")
    cls, code, _ = classify(exc)
    assert cls == "native_crash"


def test_classify_native_crash_via_texturing_failed_class():
    exc = TexturingFailed("Texture stage blew up", exit_code=3221226505)
    cls, code, _ = classify(exc)
    assert cls == "native_crash"
    assert code == 3221226505


def test_classify_oom_by_message():
    exc = RuntimeReconstructionError("CUDA error: out of memory while allocating dense maps")
    cls, _, summary = classify(exc)
    assert cls == "oom"
    assert "out of memory" in summary.lower()


def test_classify_oom_memory_allocation_phrase():
    exc = RuntimeReconstructionError("memory allocation of 8 GiB failed")
    cls, _, _ = classify(exc)
    assert cls == "oom"


def test_classify_missing_file_by_message():
    exc = RuntimeReconstructionError("No such file or directory: dense/fused.ply")
    cls, _, _ = classify(exc)
    assert cls == "missing_file"


def test_classify_missing_file_via_artifact_error():
    exc = MissingArtifactError("meshed-poisson.ply")
    cls, _, _ = classify(exc)
    assert cls == "missing_file"


def test_classify_timeout():
    exc = RuntimeReconstructionError("Subprocess timed out after 1800s")
    cls, _, _ = classify(exc)
    assert cls == "timeout"


def test_classify_unknown_default():
    exc = RuntimeReconstructionError("BA collapsed to single image; cannot proceed")
    cls, _, _ = classify(exc)
    assert cls == "unknown"


def test_classify_summary_truncated_to_240_chars():
    exc = RuntimeReconstructionError("x" * 1000)
    _, _, summary = classify(exc)
    assert len(summary) <= 240
