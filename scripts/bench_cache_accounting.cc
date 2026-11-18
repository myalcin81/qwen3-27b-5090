// CPU bookkeeping microbenchmark, not an end-to-end inference throughput claim.
// c++ -O2 -std=c++20 -Iengine/src scripts/bench_cache_accounting.cc -o /tmp/bench-cache-accounting
#include <chrono>
#include <cstdio>
#include "prefix_cache.h"
using namespace q27;
int main() {
  PagePool pool;
  pool.init(10545);
  for (int p = 1; p < pool.pages(); ++p) if (pool.alloc() != p) return 1;
  std::vector<Checkpoint> checkpoints(172);
  for (int c = 0; c < 172; ++c) for (int i = 0; i < 2440; ++i) {
    const int p = 1 + (c * 61 + i) % (pool.pages() - 1);
    checkpoints[c].pages.push_back(p);
    pool.checkpoint_ref(p);
  }
  for (int p = 1; p < pool.pages(); ++p) pool.unref(p);
  constexpr int iterations = 2000;
  int64_t totals[2] = {};
  double ms[2] = {};
  for (int mode = 0; mode < 2; ++mode) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      const int p = 1 + i % (pool.pages() - 1);
      pool.ref(p); // classification changes on every query, as with an active/restore owner
      if (mode == 0) {
        const auto refs = checkpoint_refs(checkpoints, pool.pages());
        int only = 0;
        for (int q = 1; q < pool.pages(); ++q) only += refs[q] > 0 && refs[q] == pool.refs(q);
        if (only != pool.checkpoint_only_pages()) return 2;
        totals[mode] += only;
      } else {
        totals[mode] += pool.checkpoint_only_pages();
      }
      pool.unref(p);
    }
    ms[mode] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  std::printf("{\"queries\":%d,\"checkpoint_references\":%zu,\"full_scan_ms\":%.3f,"
              "\"incremental_ms\":%.3f,\"equal\":%s}\n", iterations, size_t(172) * 2440,
              ms[0], ms[1], totals[0] == totals[1] ? "true" : "false");
  return totals[0] != totals[1];
}
