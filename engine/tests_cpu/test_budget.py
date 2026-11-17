"""Compile the real CUDA-free budget policy and compare it with the frontend planner."""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "frontend"), str(ROOT / "reference")]
import launch

GiB = 1 << 30


@pytest.fixture(scope="module")
def budget_runner(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("budget")
    source = out / "budget_test.cc"
    source.write_text(r'''
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "budget.h"
namespace q27 { [[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); } }
int main(int argc, char** argv) {
  try {
    if (argc == 2) { std::cout << q27::kernel_uses_wddm_budget(argv[1]); return 0; }
    const auto b = q27::budget_for(std::strtoull(argv[1], nullptr, 10), std::strtoull(argv[2], nullptr, 10),
                                    std::atoi(argv[3]) != 0);
    std::cout << b.budget << " " << b.asked;
  } catch (const std::exception& e) { std::cerr << e.what(); return 2; }
}
''')
    binary = out / "budget"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", *flags, "-I", str(ROOT / "engine/src"), str(source),
                    str(ROOT / "engine/src/budget.cc"), "-o", str(binary)], check=True, capture_output=True)
    return binary


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("VRAM_MARGIN_GIB", "WDDM_MARGIN_GIB", "VRAM_BUDGET_GIB"):
        monkeypatch.delenv(key, raising=False)


def native_budget(binary, free, total, wddm):
    return subprocess.run([str(binary), str(free), str(total), str(int(wddm))], capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("wddm", [False, True])
@pytest.mark.parametrize("free_gib,total_gib", [(30.19, 31.40), (30.20, 31.82), (10, 32), (0.2, 32), (40, 32)])
def test_budget_matches_frontend(budget_runner, monkeypatch, wddm, free_gib, total_gib):
    monkeypatch.setattr(launch, "uses_wddm_budget", lambda: wddm)
    free, total = int(free_gib * GiB), int(total_gib * GiB)
    result = native_budget(budget_runner, free, total, wddm)
    assert result.returncode == 0, result.stderr
    budget, asked = map(int, result.stdout.split())
    assert budget == launch.auto_budget(free, total) and asked == 0
    if (free_gib, total_gib) == (30.19, 31.40):
        assert abs(budget / GiB - (29.09 if wddm else 29.59)) < 0.001


@pytest.mark.parametrize("kernel,wddm", [("6.12.101+deb13-amd64", False), ("6.18.54.1-microsoft-standard-WSL2", True),
                                         ("Linux version 5.15.167.4-Microsoft", True), ("custom-WSL2", True), ("", True)])
def test_wsl_detection(budget_runner, monkeypatch, kernel, wddm):
    result = subprocess.run([str(budget_runner), kernel], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and bool(int(result.stdout)) == wddm
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setattr(launch.pathlib.Path, "read_text", lambda _, **kw: kernel)
    assert launch.uses_wddm_budget() == wddm


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_windows_or_unknown_platform_keeps_cap(monkeypatch, platform):
    monkeypatch.setattr(launch.sys, "platform", platform)
    assert launch.uses_wddm_budget()


def test_unreadable_linux_kernel_keeps_cap(monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "linux")
    def unreadable(_, **kw):
        raise PermissionError("no proc access")
    monkeypatch.setattr(launch.pathlib.Path, "read_text", unreadable)
    assert launch.uses_wddm_budget()


@pytest.mark.parametrize("wddm", [False, True])
@pytest.mark.parametrize("asked_gib", [20, 100])
def test_explicit_budget_only_lowers(budget_runner, monkeypatch, wddm, asked_gib):
    monkeypatch.setenv("VRAM_BUDGET_GIB", str(asked_gib))
    monkeypatch.setattr(launch, "uses_wddm_budget", lambda: wddm)
    free, total = int(30.19 * GiB), int(31.40 * GiB)
    result = native_budget(budget_runner, free, total, wddm)
    assert result.returncode == 0, result.stderr
    budget, asked = map(int, result.stdout.split())
    assert budget == min(asked_gib * GiB, launch.auto_budget(free, total))
    assert asked == (100 * GiB if asked_gib == 100 else 0)


def test_new_margin_overrides_legacy_alias(budget_runner, monkeypatch):
    monkeypatch.setenv("WDDM_MARGIN_GIB", "1.5")
    monkeypatch.setattr(launch, "uses_wddm_budget", lambda: False)
    for new_value, expected in [(None, 28.5), ("", 28.5), ("1", 29)]:
        if new_value is not None:
            monkeypatch.setenv("VRAM_MARGIN_GIB", new_value)
        result = native_budget(budget_runner, 30 * GiB, 32 * GiB, False)
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.split()[0]) == launch.auto_budget(30 * GiB, 32 * GiB) == expected * GiB


@pytest.mark.parametrize("name", ["VRAM_MARGIN_GIB", "WDDM_MARGIN_GIB", "VRAM_BUDGET_GIB"])
@pytest.mark.parametrize("value", ["nan", "inf", "-1", "bad", "1e100", "1oops"])
def test_invalid_budget_values_fail_cleanly(budget_runner, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    result = native_budget(budget_runner, 30 * GiB, 32 * GiB, False)
    assert result.returncode == 2 and name in result.stderr and "Sanitizer" not in result.stderr
    if "MARGIN" in name:
        with pytest.raises(ValueError):
            launch.margin_gib()
