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
#include <cassert>
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
    checkpoint_ref_.assign(pages, 0);
    checkpoint_only_ = 0;
    ref_[0] = 1;
    free_.clear();
    for (int p = 1; p < pages; p++) free_.insert(free_.end(), p);
    high_ = high;
  }
  int pages() const { return int(ref_.size()); }
  int free_pages() const { return int(free_.size()); }
  int refs(int p) const { return ref_[p]; }
  int checkpoint_refs(int p) const { return checkpoint_ref_[p]; }
  int checkpoint_only_pages() const { return checkpoint_only_; }
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
    assert(p != q && ref_[q] == 1 && checkpoint_ref_[q] == 0);
    checkpoint_only_ -= checkpoint_only(p) + checkpoint_only(q);
    ref_[q] = ref_[p];
    checkpoint_ref_[q] = checkpoint_ref_[p];
    ref_[p] = 0;
    checkpoint_ref_[p] = 0;
    checkpoint_only_ += checkpoint_only(q);
    free_.insert(p);
  }
  void ref(int p) { change_refs(p, 1, 0); }
  void unref(int p) { change_refs(p, -1, 0); }
  // A checkpoint owns both a total reference and a checkpoint reference. Active/parked sequences and
  // in-flight copies use ref/unref, so they immediately remove a page from the reclaimable count.
  void checkpoint_ref(int p) { change_refs(p, 1, 1); }
  void checkpoint_unref(int p) { change_refs(p, -1, -1); }

 private:
  bool checkpoint_only(int p) const { return checkpoint_ref_[p] > 0 && checkpoint_ref_[p] == ref_[p]; }
  void change_refs(int p, int all, int checkpoints) {
    assert(ref_[p] > 0);
    checkpoint_only_ -= checkpoint_only(p);
    ref_[p] += all;
    checkpoint_ref_[p] += checkpoints;
    assert(checkpoint_ref_[p] >= 0 && checkpoint_ref_[p] <= ref_[p]);
    checkpoint_only_ += checkpoint_only(p);
    if (!ref_[p]) free_.insert(p);
  }
  std::vector<int32_t> ref_, checkpoint_ref_;
  int checkpoint_only_ = 0;
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
  uint64_t last_reuse = 0;      // nonzero: bounded protection earned by a successful prefix restore
  int pos() const { return int(ids.size()); }
};

// Protection is charged conservatively by logical pages, including GPU-resident/shared pages. It therefore
// cannot silently exceed the host budget when a protected prefix is demoted or another owner disappears.
// Also bound state slots: a scan must always have probationary space. Dropping protection does not drop data.
inline void trim_protected(std::vector<Checkpoint>& checkpoints, size_t page_budget, size_t slot_budget) {
  size_t pages = 0, slots = 0;
  std::vector<Checkpoint*> protected_entries;
  for (auto& c : checkpoints) if (c.last_reuse) {
    pages += c.pages.size();
    ++slots;
    protected_entries.push_back(&c);
  }
  std::stable_sort(protected_entries.begin(), protected_entries.end(), [](const auto* a, const auto* b) {
    return a->last_reuse < b->last_reuse;
  });
  for (auto* c : protected_entries) {
    if (pages <= page_budget && slots <= slot_budget) break;
    pages -= c->pages.size();
    --slots;
    c->last_reuse = 0;
  }
}

inline void protect_checkpoint(std::vector<Checkpoint>& checkpoints, size_t index, uint64_t clock,
                                size_t page_budget, size_t slot_budget) {
  auto& c = checkpoints[index];
  // An oversized prefix must not flush the protected set only to lose its own protection afterwards.
  if (slot_budget && c.pages.size() <= page_budget) c.last_reuse = clock;
  trim_protected(checkpoints, page_budget, slot_budget);
}

inline bool eviction_before(const Checkpoint& a, const Checkpoint& b) {
  if (bool(a.last_reuse) != bool(b.last_reuse)) return !a.last_reuse;
  return a.last_use < b.last_use;
}

inline int cache_eviction_index(const std::vector<Checkpoint>& checkpoints, bool host_only = false) {
  int victim = -1;
  for (size_t i = 0; i < checkpoints.size(); ++i) {
    const auto& c = checkpoints[i];
    if (host_only && std::none_of(c.pages.begin(), c.pages.end(), [](int p) { return p < 0; })) continue;
    if (victim < 0 || eviction_before(c, checkpoints[victim])) victim = int(i);
  }
  return victim;
}

struct CacheStats {
  int64_t lookups = 0, hits = 0, prompt_tokens = 0, cached_tokens = 0, saved = 0, evicted = 0;
  int64_t demoted = 0, restored = 0, evicted_state = 0, evicted_gpu = 0, evicted_host = 0;
};

struct CacheInventory {
  int64_t unique_tokens = 0, logical_full_pages = 0, physical_full_pages = 0;
  int64_t page_references = 0, gpu_pages = 0, host_pages = 0, leaves = 0;
};

// Token-prefix union, not sum(checkpoint lengths): ancestors and shared branches count only once. This is
// diagnostic accounting, never a substitute for exact lookup or a license to deduplicate numerical state.
inline CacheInventory cache_inventory(const std::vector<Checkpoint>& checkpoints, int page_tokens) {
  CacheInventory out;
  std::vector<const Checkpoint*> sorted;
  std::set<int32_t> gpu, host, full;
  for (const auto& c : checkpoints) {
    sorted.push_back(&c);
    out.page_references += int64_t(c.pages.size());
    for (size_t i = 0; i < c.pages.size(); ++i) {
      const int p = c.pages[i];
      if (p > 0) gpu.insert(p);
      if (p < 0) host.insert(p);
      if (p && (i + 1) * size_t(page_tokens) <= c.ids.size()) full.insert(p);
    }
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) { return a->ids < b->ids; });
  const Checkpoint* previous = nullptr;
  for (const auto* c : sorted) {
    size_t shared = 0;
    if (previous) {
      const size_t end = std::min(previous->ids.size(), c->ids.size());
      while (shared < end && previous->ids[shared] == c->ids[shared]) ++shared;
      if (shared < previous->ids.size()) ++out.leaves;
    }
    out.unique_tokens += int64_t(c->ids.size() - shared);
    out.logical_full_pages += c->pos() / page_tokens - int64_t(shared / page_tokens);
    previous = c;
  }
  if (previous) ++out.leaves;
  out.gpu_pages = int64_t(gpu.size());
  out.host_pages = int64_t(host.size());
  out.physical_full_pages = int64_t(full.size());
  return out;
}

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
    if (rank < best_rank || (rank == best_rank && eviction_before(c, *nodes[victim]))) {
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
  const Checkpoint* oldest = nullptr;
  for (const auto& c : checkpoints) {
    if (oldest && c.last_use >= oldest->last_use) continue;
    for (int p : c.pages)
      if (p > 0 && pool.checkpoint_refs(p) == pool.refs(p)) { oldest = &c; break; }
  }
  std::vector<int32_t> pages;
  if (oldest)
    for (int p : oldest->pages) {
      if (p > 0 && pool.checkpoint_refs(p) == pool.refs(p)) pages.push_back(p);
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
