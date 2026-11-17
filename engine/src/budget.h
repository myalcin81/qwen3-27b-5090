// VRAM policy, independent of CUDA so platform/default compatibility can be tested on a CPU runner.
#pragma once

#include <cstddef>
#include <string>

namespace q27 {

struct Budget {
  size_t free_at_start = 0;
  size_t total = 0;
  size_t budget = 0;
  size_t asked = 0;  // requested VRAM_BUDGET_GIB when above the automatic budget
};

bool kernel_uses_wddm_budget(std::string release);
bool uses_wddm_budget();
Budget budget_for(size_t free, size_t total, bool wddm = uses_wddm_budget());

}  // namespace q27
