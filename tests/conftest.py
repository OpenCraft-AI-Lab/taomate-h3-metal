# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Build the macOS test helper in an isolated temporary directory."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def native_memory_watcher(tmp_path_factory):
    if sys.platform != "darwin":
        pytest.skip("native macOS watcher")
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("requires Xcode Command Line Tools")
    source = Path(__file__).resolve().parents[1] / "scripts/memory_sample.c"
    binary = tmp_path_factory.mktemp("memory-watcher") / "memory-sample"
    subprocess.run(
        [compiler, "-std=c11", "-D_DARWIN_C_SOURCE", "-O2", "-Wall", "-Wextra",
         str(source), "-o", str(binary)],
        check=True,
    )
    return binary
