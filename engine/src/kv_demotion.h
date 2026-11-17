// Ownership of a single in-flight GPU -> host KV batch. CUDA event ordering belongs to Model; none of
// these references may be released/published until the copy stream's completion fence has been observed.
#pragma once

#include <cassert>
#include <map>

#include "host_kv.h"
#include "prefix_cache.h"

namespace q27 {

class KvDemotion {
 public:
  bool empty() const { return pages_.empty(); }
  size_t size() const { return pages_.size(); }
  const std::map<int, int>& pages() const { return pages_; }

  // host is a fresh allocation carrying its transfer reference. If insertion throws, the caller still
  // owns that allocation. A successful hold also pins the GPU source even if its last checkpoint dies.
  void hold(int gpu, int host, PagePool& pool) {
    assert(gpu > 0 && host < 0 && pool.refs(gpu) > 0 && !pages_.contains(gpu));
    pages_.emplace(gpu, host);
    pool.ref(gpu);
  }

  template <class ReleaseCheckpoint>
  void publish_after_fence(std::vector<Checkpoint>& checkpoints, ReleaseCheckpoint release_checkpoint,
                           HostKvPages& host) {
    rebind_checkpoints(checkpoints, pages_, release_checkpoint, [&](int h) { host.ref(h); });
  }

  // Also used after a queueing error, but only after draining the copy stream. Unpublished targets and
  // sources whose last checkpoint was evicted then become free; active sequence owners remain intact.
  void release_after_fence(PagePool& pool, HostKvPages& host) {
    for (const auto& [p, h] : pages_) {
      pool.unref(p);
      host.unref(h);
    }
    pages_.clear();
  }

 private:
  std::map<int, int> pages_;
};

}  // namespace q27
