"""Exercise production host-page ownership and tier transitions without CUDA, with optional sanitizers."""

import os
import pathlib
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def host_kv_test(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("host-kv")
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    source = out / "host_kv.cc"
    source.write_text(r'''
#include <cassert>
#include <cstring>
#include <string>
#include "host_kv.h"
#include "prefix_cache.h"
using namespace q27;
int main(int argc, char** argv) {
  const std::string test = argv[1];
  int allocations = 0, frees = 0;
  auto alloc = [&](size_t n) { allocations++; return new char[n]; };
  auto release = [&](char* p) { frees++; delete[] p; };
  if (test == "bounded") {
    { HostKvPages h;
      h.init(19, 32, alloc, release);
      assert(h.bytes() == 0 && h.budget() == 608 && h.used() == 0);
      std::vector<int> pages;
      for (int i = 0; i < 19; i++) { pages.push_back(h.alloc()); assert(pages.back() < 0); }
      assert(h.alloc() == 0 && h.used() == 19 && h.bytes() == 608 && allocations == 2);
      h.ref(pages[0]); h.unref(pages[0]);
      assert(h.alloc() == 0);  // one owner still holds it
      h.unref(pages[0]);
      assert(h.alloc() == pages[0] && allocations == 2);
    }
    assert(frees == allocations);
  } else if (test == "disabled") {
    HostKvPages h; h.init(0, 32, alloc, release);
    assert(h.alloc() == 0 && allocations == 0 && h.bytes() == 0);
  } else if (test == "allocation_failure") {
    HostKvPages h;
    h.init(32, 32, [&](size_t n) -> char* { return allocations ? nullptr : alloc(n); }, release);
    std::vector<int> pages;
    for (int i = 0; i < 16; i++) pages.push_back(h.alloc());
    assert(h.alloc() == 0 && h.failures() == 1 && h.allocated() == 16);
    h.unref(pages.back());  // failure must not disable reuse of the previously allocated tier
    assert(h.alloc() == pages.back() && h.failures() == 1);
  } else if (test == "shared_round_trip") {
    HostKvPages h; h.init(2, 64, alloc, release);
    PagePool gpu; gpu.init(3);
    int p = gpu.alloc(); gpu.ref(p);  // two checkpoint owners
    std::vector<Checkpoint> cs(2);
    cs[0].pages = {p}; cs[0].slot = 4; cs[0].ring = true; cs[0].last_use = 1;
    cs[1].pages = {p}; cs[1].slot = 5; cs[1].ring = true; cs[1].last_use = 2;
    auto batch = demotion_batch(cs, gpu);
    assert(batch == std::vector<int32_t>{p});
    const int hp = h.alloc();
    std::memset(h.data(hp), 73, 64);  // completed D2H copy (GPU runner tests the real transfer)
    rebind_checkpoints(cs, {{p, hp}}, [&](int x) { gpu.unref(x); }, [&](int x) { h.ref(x); });
    h.unref(hp);
    assert(h.used() == 1 && h.refs(hp) == 2 && gpu.refs(p) == 0);
    assert(cs[0].slot == 4 && cs[1].slot == 5 && cs[0].ring && cs[1].ring);
    assert(checkpoint_refs(cs, gpu.pages())[p] == 0 && demotion_batch(cs, gpu).empty());
    const int reused = gpu.alloc(); assert(reused == p); // GPU reuse cannot alias a host checkpoint
    const int restored = gpu.alloc();
    char data[64]; std::memcpy(data, h.data(hp), 64);
    h.ref(hp); // restore's temporary reference, held until the copy has finished
    rebind_checkpoints(cs, {{hp, restored}}, [&](int x) { h.unref(x); }, [&](int x) { gpu.ref(x); });
    h.unref(hp);
    for (char c : data) assert(c == 73);
    assert(h.used() == 0 && gpu.refs(restored) == 3 && gpu.refs(reused) == 1);
    assert(cs[0].pages == cs[1].pages && cs[0].pages[0] == restored);
    assert(demotion_batch(cs, gpu).empty()); // active sequence owns the third reference
    gpu.unref(restored);
    assert(demotion_batch(cs, gpu) == std::vector<int32_t>{restored});
  } else if (test == "active_and_lru") {
    PagePool gpu; gpu.init(5);
    int active = gpu.alloc(), recent = gpu.alloc(), old = gpu.alloc();
    gpu.ref(active);  // active/parked request plus checkpoint
    std::vector<Checkpoint> cs(3);
    cs[0].pages = {active, -1}; cs[0].last_use = 0;
    cs[1].pages = {recent}; cs[1].last_use = 20;
    cs[2].pages = {old}; cs[2].last_use = 10;
    assert(demotion_batch(cs, gpu) == std::vector<int32_t>{old});
    gpu.unref(active);
    assert(demotion_batch(cs, gpu) == std::vector<int32_t>{active});
  } else if (test == "batch_limit") {
    PagePool gpu; gpu.init(601);
    std::vector<Checkpoint> cs(1);
    for (int i = 0; i < 600; i++) cs[0].pages.push_back(gpu.alloc());
    assert(demotion_batch(cs, gpu).size() == 256);
  } else if (test == "restore_protection") {
    HostKvPages h; h.init(1, 32, alloc, release);
    const int hp = h.alloc();
    std::memset(h.data(hp), 91, 32);
    h.ref(hp);  // restoration retains its source before eviction may remove the checkpoint
    h.unref(hp);
    assert(h.alloc() == 0 && h.data(hp)[0] == 91);
    h.unref(hp);
    assert(h.alloc() == hp);
  } else if (test == "partial_page") {
    HostKvPages h; h.init(1, 64, alloc, release);
    const int hp = h.alloc();
    std::memset(h.data(hp), 21, 64);
    char seq[64]; std::memcpy(seq, h.data(hp), 64);
    seq[5] = 42;  // partial-page restore is private; checkpoint remains unchanged
    assert(h.data(hp)[5] == 21 && h.refs(hp) == 1);
  } else return 1;
}
''')
    binary = out / "host-kv"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", *flags, "-I", str(src), str(source), "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("case", ["bounded", "disabled", "allocation_failure", "shared_round_trip", "active_and_lru",
                                  "batch_limit", "restore_protection", "partial_page"])
def test_host_page_ownership(host_kv_test, case):
    result = subprocess.run([str(host_kv_test), case], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
