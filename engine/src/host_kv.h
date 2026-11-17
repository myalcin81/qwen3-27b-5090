// Bounded, lazily allocated storage for immutable KV pages. No CUDA dependency: the caller supplies
// pinned allocation/free functions and orders transfers before releasing or publishing a page.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace q27 {

class HostKvPages {
 public:
  using Alloc = std::function<char*(size_t)>;
  using Free = std::function<void(char*)>;
  HostKvPages() = default;
  HostKvPages(const HostKvPages&) = delete;
  HostKvPages& operator=(const HostKvPages&) = delete;
  ~HostKvPages() { for (char* p : slabs_) free_mem_(p); }

  void init(int capacity, size_t page_bytes, Alloc alloc, Free free_mem) {
    assert(slabs_.empty());
    capacity_ = capacity;
    page_bytes_ = page_bytes;
    alloc_ = std::move(alloc);
    free_mem_ = std::move(free_mem);
  }
  int capacity() const { return capacity_; }
  int allocated() const { return int(pages_.size()); }
  int used() const { return allocated() - int(free_.size()); }
  size_t bytes() const { return size_t(allocated()) * page_bytes_; }
  size_t budget() const { return size_t(capacity_) * page_bytes_; }
  int failures() const { return failures_; }

  // Negative references cannot alias a recycled GPU page number. Slot zero is encoded as -1.
  static int encode(int slot) { return -slot - 1; }
  static int decode(int page) { assert(page < 0); return -page - 1; }
  char* data(int page) const { return pages_[decode(page)].data; }
  int refs(int page) const { return pages_[decode(page)].refs; }
  void ref(int page) { assert(refs(page) > 0); pages_[decode(page)].refs++; }
  void unref(int page) {
    const int slot = decode(page);
    assert(pages_[slot].refs > 0);
    if (--pages_[slot].refs == 0) free_.push_back(slot);
  }
  // Allocates at most 16 pages at a time. A pinning failure leaves existing pages usable and disables
  // further growth for this engine instance; callers fall back to eviction, never unbounded RAM use.
  int alloc() {
    if (free_.empty() && allocated() < capacity_ && failures_ == 0) {
      const int n = std::min(16, capacity_ - allocated()), first = allocated();
      pages_.reserve(size_t(first) + n);
      free_.reserve(size_t(first) + n);
      slabs_.reserve(slabs_.size() + 1);
      char* mem = alloc_(size_t(n) * page_bytes_);
      if (!mem) { failures_++; return 0; }
      slabs_.push_back(mem);
      for (int i = 0; i < n; i++) {
        pages_.push_back({mem + size_t(i) * page_bytes_, 0});
        free_.push_back(first + i);
      }
    }
    if (free_.empty()) return 0;  // no host page (GPU page zero is reserved too)
    const int slot = free_.back();
    free_.pop_back();
    pages_[slot].refs = 1;
    return encode(slot);
  }

 private:
  struct Page { char* data; int refs; };
  int capacity_ = 0, failures_ = 0;
  size_t page_bytes_ = 0;
  std::vector<Page> pages_;
  std::vector<int> free_;
  std::vector<char*> slabs_;
  Alloc alloc_;
  Free free_mem_;
};

}  // namespace q27
