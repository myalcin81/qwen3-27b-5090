// Prefix cache bookkeeping (host side, docs/architecture.md §9): the KV page pool and the state checkpoints.
//
// A GDN layer can only resume from a saved state, so the cache unit is a checkpoint: the token prefix
// [0, pos), the KV pages covering it and a copy of the recurrent state (GDN + conv + the draft's
// context ring) in a pinned host slot. KV pages past the deepest checkpoint of a prefix are useless
// (the layers have to run over those tokens again anyway), so pages are kept only through the
// checkpoints that reference them. Checkpoints of one conversation share their full pages by
// reference count; the last, partial page is copied before the resumed sequence writes into it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <vector>

namespace q27 {

class PagePool {
 public:
  // Pages 1 .. pages - 1 are handed out; page 0 is the scratch page of unmapped page table entries. The
  // lowest free page goes first, so the parking areas at the top of the pool stay free while there is room
  // (P7-13); high (Q27_PAGES_HIGH, tests): the highest, so parking moves pages.
  void init(int pages, bool high = false) {
    ref_.assign(pages, 0);
    ref_[0] = 1;
    free_.clear();
    for (int p = 1; p < pages; p++) free_.insert(free_.end(), p);
    high_ = high;
  }
  int pages() const { return int(ref_.size()); }
  int free_pages() const { return int(free_.size()); }
  int refs(int p) const { return ref_[p]; }
  // -1 when the pool is empty.
  int alloc() { return alloc_outside(0, 0); }
  // A free page outside [lo, hi); -1 when there is none.
  int alloc_outside(int lo, int hi) {
    int p = -1;
    if (!high_) {
      auto it = free_.begin();
      if (it != free_.end() && *it >= lo && *it < hi) it = free_.lower_bound(hi);
      if (it != free_.end()) p = *it;
    } else {
      auto it = free_.lower_bound(hi);
      if (it != free_.end()) p = *free_.rbegin();
      else if ((it = free_.lower_bound(lo)) != free_.begin()) p = *std::prev(it);
    }
    if (p < 0) return -1;
    free_.erase(p);
    ref_[p] = 1;
    return p;
  }
  // Free pages outside [lo, hi).
  int free_outside(int lo, int hi) const {
    return int(free_.size()) - int(std::distance(free_.lower_bound(lo), free_.lower_bound(hi)));
  }
  // Takes the free page p (a parking area's); false when it is in use.
  bool take(int p) {
    if (!free_.erase(p)) return false;
    ref_[p] = 1;
    return true;
  }
  // The references of page p move to page q (from alloc_outside); p is free then.
  void move(int p, int q) {
    ref_[q] = ref_[p];
    ref_[p] = 0;
    free_.insert(p);
  }
  void ref(int p) { ref_[p]++; }
  void unref(int p) {
    if (--ref_[p] == 0) free_.insert(p);
  }

 private:
  std::vector<int32_t> ref_;
  std::set<int32_t> free_;
  bool high_ = false;
};

struct Checkpoint {
  std::vector<int32_t> ids;    // the tokens [0, pos)
  std::vector<int32_t> pages;  // KV pages of [0, pos): >0 GPU, <0 HostKvPages (one reference each)
  int slot = -1;               // host state slot
  bool ring = false;           // the draft's context ring is valid for [pos - 2048, pos)
  bool boundary = false;       // explicitly requested prompt/shared-prefix boundary (rather than an interval)
  uint64_t last_use = 0;
  int pos() const { return int(ids.size()); }
};

struct CacheStats {
  int64_t lookups = 0, hits = 0, prompt_tokens = 0, cached_tokens = 0, saved = 0, evicted = 0;
  int64_t demoted = 0, restored = 0, evicted_state = 0, evicted_gpu = 0, evicted_host = 0;
};

// Under state-slot pressure, retire redundant points of a prefix chain before its useful ends. In particular,
// retain both the deepest prompt boundary (exact retries) and the generated continuation (the next turn).
// Branch points stay useful to multiple descendants. Classification is only an eviction heuristic: actual
// restore still compares every token and checks ring validity. No fingerprints stand in for token identity.
// -1 means an optional interval checkpoint should be skipped rather than displacing an independent prefix.
inline int state_eviction_index(const std::vector<Checkpoint>& checkpoints, const Checkpoint& pending,
                                 bool interval = false) {
  const size_t n = checkpoints.size();
  if (!n) return -1;
  std::vector<const Checkpoint*> nodes;
  for (const auto& c : checkpoints) nodes.push_back(&c);
  nodes.push_back(&pending);
  std::vector<size_t> order(n + 1);
  for (size_t i = 0; i <= n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    // Equal-token plain/spec entries are adjacent, with the ring-capable one covering the plain one.
    if (nodes[a]->ids == nodes[b]->ids) return nodes[a]->ring < nodes[b]->ring;
    return nodes[a]->ids < nodes[b]->ids;
  });
  std::vector<int> parent(n + 1, -1), children(n + 1, 0), covered(n + 1, 0), boundaries(n + 1, 0);
  std::vector<size_t> stack;
  for (size_t i : order) {
    const auto& ids = nodes[i]->ids;
    while (!stack.empty()) {
      const auto& prev = nodes[stack.back()]->ids;
      if (prev.size() <= ids.size() && std::equal(prev.begin(), prev.end(), ids.begin())) break;
      stack.pop_back();
    }
    if (!stack.empty()) {
      parent[i] = int(stack.back());
      ++children[stack.back()];
    }
    stack.push_back(i);
  }
  // Propagate the capabilities of all descendants, including the not-yet-saved incoming checkpoint.
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const size_t i = *it;
    if (const int p = parent[i]; p >= 0) {
      const int capability = nodes[i]->ring ? 3 : 1;  // plain / speculative
      covered[p] |= covered[i] | capability;
      boundaries[p] |= boundaries[i] | (nodes[i]->boundary ? capability : 0);
    }
  }
  int victim = -1, best_rank = 4;
  for (size_t i = 0; i < n; ++i) {
    const auto& c = *nodes[i];
    const int need = c.ring ? 2 : 1;
    const int rank = children[i] > 1 ? 3 : !(covered[i] & need) ? 2 :
                     !c.boundary ? 0 : (boundaries[i] & need) ? 1 : 2;
    if (rank < best_rank || (rank == best_rank && c.last_use < nodes[victim]->last_use)) {
      victim = int(i);
      best_rank = rank;
    }
  }
  return interval && best_rank >= 2 ? -1 : victim;
}

// Count only checkpoint references, so active/parked sequences (and in-flight restores) prevent demotion.
inline std::vector<int32_t> checkpoint_refs(const std::vector<Checkpoint>& checkpoints, int pages) {
  std::vector<int32_t> refs(pages, 0);
  for (const auto& c : checkpoints)
    for (int p : c.pages) if (p > 0) refs[p]++;
  return refs;
}

inline std::vector<int32_t> demotion_batch(const std::vector<Checkpoint>& checkpoints, const PagePool& pool,
                                          size_t limit = 256) {
  const auto refs = checkpoint_refs(checkpoints, pool.pages());
  const Checkpoint* oldest = nullptr;
  for (const auto& c : checkpoints) {
    if (oldest && c.last_use >= oldest->last_use) continue;
    for (int p : c.pages)
      if (p > 0 && refs[p] == pool.refs(p)) { oldest = &c; break; }
  }
  std::vector<int32_t> pages;
  if (oldest)
    for (int p : oldest->pages) {
      if (p > 0 && refs[p] == pool.refs(p)) pages.push_back(p);
      if (pages.size() == limit) break;
    }
  return pages;
}

// Transfer all checkpoint references together, after the copy fence. The caller retains a temporary
// reference until this finishes, so the destination cannot be recycled midway through a shared remap.
template <class Release, class Retain>
void rebind_checkpoints(std::vector<Checkpoint>& checkpoints, const std::map<int, int>& moved,
                        Release release, Retain retain) {
  for (auto& c : checkpoints)
    for (int32_t& p : c.pages)
      if (const auto it = moved.find(p); it != moved.end()) {
        release(p);
        p = it->second;
        retain(p);
      }
}

}  // namespace q27
