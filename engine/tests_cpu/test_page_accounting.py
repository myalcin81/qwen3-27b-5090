"""Compare the production incremental counters to an independent ownership scan after every mutation."""

import os
import pathlib
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def accounting_test(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("page-accounting")
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    source = out / "accounting.cc"
    source.write_text(r'''
#include <cassert>
#include <random>
#include <string>
#include "prefix_cache.h"
using namespace q27;
int main(int argc, char** argv) {
  const bool high = std::string(argv[1]) == "high";
  PagePool pool;
  pool.init(97, high);
  std::vector<int32_t> active;
  std::vector<Checkpoint> checkpoints;
  std::mt19937 rng(20261118);
  auto audit = [&] {
    std::vector<int> all(97), cached(97);
    all[0] = 1; // scratch page
    for (int p : active) ++all[p];
    for (const auto& c : checkpoints) for (int p : c.pages) { ++all[p]; ++cached[p]; }
    int only = 0, free = 0;
    for (int p = 0; p < 97; ++p) {
      assert(pool.refs(p) == all[p]);
      assert(pool.checkpoint_refs(p) == cached[p]);
      only += cached[p] > 0 && all[p] == cached[p];
      free += all[p] == 0;
    }
    assert(pool.checkpoint_only_pages() == only && pool.free_pages() == free);
    for (int p : demotion_batch(checkpoints, pool)) assert(cached[p] > 0 && all[p] == cached[p]);
  };
  for (int step = 0; step < 12000; ++step) {
    const int op = rng() % 7;
    if (op == 0) {
      const int p = pool.alloc();
      if (p >= 0) active.push_back(p);
    } else if (op == 1 && !active.empty()) {
      Checkpoint c;
      c.last_use = step;
      for (int k = 0, n = 1 + rng() % 6; k < n; ++k) {
        const int p = active[rng() % active.size()];
        c.pages.push_back(p); pool.checkpoint_ref(p);
      }
      checkpoints.push_back(std::move(c));
    } else if (op == 2 && !checkpoints.empty()) {
      const int i = rng() % checkpoints.size();
      for (int p : checkpoints[i].pages) pool.checkpoint_unref(p);
      checkpoints.erase(checkpoints.begin() + i);
    } else if (op == 3 && !active.empty()) {
      const int i = rng() % active.size();
      pool.unref(active[i]); active.erase(active.begin() + i);
    } else if (op == 4 && !checkpoints.empty()) {
      // Temporary restore owner must immediately protect a checkpoint-only page from demotion.
      const auto& c = checkpoints[rng() % checkpoints.size()];
      const int p = c.pages[rng() % c.pages.size()];
      pool.ref(p); active.push_back(p);
    } else if (op == 5 && (!active.empty() || !checkpoints.empty())) {
      // The real parking/VMM move rebinds every owner and transfers the checkpoint count with it.
      const int p = active.empty() ? checkpoints.front().pages.front() : active[rng() % active.size()];
      const int q = pool.alloc_outside(p, p + 1);
      if (q >= 0) {
        pool.move(p, q);
        for (auto& page : active) if (page == p) page = q;
        for (auto& c : checkpoints) for (auto& page : c.pages) if (page == p) page = q;
        assert(pool.take(p)); active.push_back(p); // parking area's reservation is not a checkpoint
      }
    } else if (op == 6 && !active.empty()) {
      const int p = active[rng() % active.size()];
      pool.ref(p); active.push_back(p);
    }
    audit();
  }
  for (const auto& c : checkpoints) for (int p : c.pages) pool.checkpoint_unref(p);
  checkpoints.clear();
  for (int p : active) pool.unref(p);
  active.clear();
  audit();
  assert(pool.free_pages() == 96 && pool.checkpoint_only_pages() == 0);
  pool.init(97, !high);
  audit();
}
''', encoding="utf-8")
    binary = out / "accounting"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", *flags, "-I", str(src), str(source), "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("order", ["low", "high"])
def test_incremental_accounting_matches_full_scan(accounting_test, order):
    result = subprocess.run([str(accounting_test), order], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()
