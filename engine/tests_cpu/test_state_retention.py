"""Compile the actual state-slot eviction policy with sanitizers; no CUDA or weights required."""

import os
import pathlib
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def retention_test(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("state-retention")
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    source = out / "retention.cc"
    source.write_text(r'''
#include <cassert>
#include <string>
#include "prefix_cache.h"
using namespace q27;
Checkpoint ck(std::vector<int32_t> ids, bool boundary, uint64_t use, bool ring = true) {
  Checkpoint c; c.ids = std::move(ids); c.boundary = boundary; c.last_use = use; c.ring = ring; return c;
}
int main(int argc, char** argv) {
  const std::string test = argv[1];
  const auto next = ck({9, 9}, true, 99);
  if (test == "intermediate_before_lru") {
    std::vector<Checkpoint> cs = {ck({8}, true, 1), ck({1,2}, false, 4),
                                 ck({1,2,3}, true, 5), ck({1,2,3,4}, false, 6)};
    assert(state_eviction_index(cs, next) == 1); // keep older independent prefix and both useful ends
  } else if (test == "covered_boundary") {
    std::vector<Checkpoint> cs = {ck({8}, true, 1), ck({1,2}, true, 4),
                                 ck({1,2,3}, true, 5), ck({1,2,3,4}, false, 6)};
    assert(state_eviction_index(cs, next) == 1); // earlier boundary, not the prompt retry point
  } else if (test == "branch_point") {
    std::vector<Checkpoint> cs = {ck({1}, true, 1), ck({1,2}, true, 4), ck({1,3}, true, 5)};
    assert(state_eviction_index(cs, next) == 1); // common boundary has two children; keep it
  } else if (test == "pending_branch") {
    std::vector<Checkpoint> cs = {ck({1}, true, 1), ck({1,2}, true, 4), ck({8}, true, 3)};
    assert(state_eviction_index(cs, ck({1,3}, true, 9)) == 2);
  } else if (test == "ring_compatibility") {
    std::vector<Checkpoint> cs = {ck({8}, true, 1), ck({1,2}, false, 4), ck({1,2,3}, false, 5, false)};
    assert(state_eviction_index(cs, next) == 0); // plain descendant cannot replace a speculative checkpoint
    cs[2].ring = true;
    assert(state_eviction_index(cs, next) == 1);
  } else if (test == "equal_tokens_ring_upgrade") {
    std::vector<Checkpoint> cs = {ck({8}, true, 1), ck({1,2}, true, 8, false)};
    assert(state_eviction_index(cs, ck({1,2}, true, 9)) == 1);
  } else if (test == "interval_backoff") {
    std::vector<Checkpoint> cs = {ck({1,2}, true, 1), ck({1,2,3}, false, 4), ck({8}, true, 3)};
    assert(state_eviction_index(cs, next, true) == -1); // no useful state is sacrificed for an interval
    assert(state_eviction_index(cs, next, false) == 0); // ordinary checkpoint uses LRU as the final fallback
  } else if (test == "exact_prefix_and_ties") {
    std::vector<Checkpoint> cs = {ck({1,2,3}, false, 5), ck({1,2,4,5}, false, 5)};
    assert(state_eviction_index(cs, next) == 0); // differing token is not a covered ancestor
    assert(state_eviction_index({}, next) == -1);
  } else if (test == "forty_sessions") {
    for (bool legacy : {true, false}) {
      std::vector<Checkpoint> cs;
      uint64_t clock = 0;
      int hits = 0;
      auto save = [&](Checkpoint c, bool interval) {
        for (auto& old : cs) if (old.ids == c.ids) {
          old.last_use = ++clock; old.boundary |= c.boundary; return;
        }
        if (cs.size() == 86) {
          int v = legacy ? int(std::min_element(cs.begin(), cs.end(), [](const auto& a, const auto& b) {
            return a.last_use < b.last_use;
          }) - cs.begin()) : state_eviction_index(cs, c, interval);
          if (v < 0) return;
          cs.erase(cs.begin() + v);
        }
        c.last_use = ++clock; cs.push_back(std::move(c));
        assert(cs.size() <= 86);
      };
      save(ck({1}, true, 0), false);
      for (int round = 0; round < 4; ++round) for (int session = 0; session < 40; ++session) {
        std::vector<int32_t> ids(15, session + 10); ids[0] = 1;
        if (round) for (const auto& c : cs)
          if (c.pos() == 12 && std::equal(c.ids.begin(), c.ids.end(), ids.begin())) { ++hits; break; }
        for (int pos : {2, 4, 6, 8, 10, 12, 14})
          save(ck(std::vector<int32_t>(ids.begin(), ids.begin() + pos), pos == 2 || pos == 12, 0),
               pos == 4 || pos == 6 || pos == 8 || pos == 10);
      }
      // 86 slots, 40 distinct chains, four interval snapshots plus prompt/continuation points on each.
      // Legacy LRU loses every retried prompt. Pressure compaction retains all 120 warm retry points.
      assert(hits == (legacy ? 0 : 120));
    }
  } else return 1;
}
''', encoding="utf-8")
    binary = out / "retention"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", *flags, "-I", str(src), str(source), "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("case", ["intermediate_before_lru", "covered_boundary", "branch_point", "pending_branch",
                                  "ring_compatibility", "equal_tokens_ring_upgrade", "interval_backoff",
                                  "exact_prefix_and_ties", "forty_sessions"])
def test_state_slot_retention(retention_test, case):
    result = subprocess.run([str(retention_test), case], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
