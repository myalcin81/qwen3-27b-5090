#include "budget.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "check.h"

namespace q27 {
namespace {
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

double env_gib(const char* name, double fallback) {
  const char* value = getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  const double n = std::strtod(value, &end);
  const bool parsed = end != value;
  while (end && std::isspace(static_cast<unsigned char>(*end))) end++;
  Q27_CHECK(parsed && end && !*end && std::isfinite(n) && n >= 0 &&
              n < double(std::numeric_limits<size_t>::max()) / kGiB,
            std::string(name) + " must be a finite, nonnegative GiB value");
  return n;
}
}  // namespace

bool kernel_uses_wddm_budget(std::string release) {
  std::transform(release.begin(), release.end(), release.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  // An unknown kernel keeps the old conservative ceiling. Docker on WSL reports the WSL host kernel too.
  return release.empty() || release.find("microsoft") != std::string::npos || release.find("wsl") != std::string::npos;
}

bool uses_wddm_budget() {
#if defined(__linux__) && !defined(_WIN32)
  for (const char* path : {"/proc/sys/kernel/osrelease", "/proc/version"}) {
    std::ifstream input(path);
    std::string release;
    if (std::getline(input, release) && !release.empty()) return kernel_uses_wddm_budget(release);
  }
#endif
  return true;  // Windows and unknown platforms retain the existing WDDM safety ceiling.
}

Budget budget_for(size_t free, size_t total, bool wddm) {
  Budget b;
  b.free_at_start = free = std::min(free, total);
  b.total = total;
  const char* margin_name = getenv("VRAM_MARGIN_GIB");
  const double margin = margin_name && *margin_name ? env_gib("VRAM_MARGIN_GIB", 0.6) : env_gib("WDDM_MARGIN_GIB", 0.6);
  double available = double(free) / kGiB - margin;
  if (wddm) available = std::min(available, 30.3 - double(total - free) / kGiB);
  b.budget = size_t(std::max(0.0, available) * kGiB);
  // Explicit budgets only lower the automatic bound on every platform.
  const size_t asked = size_t(env_gib("VRAM_BUDGET_GIB", double(b.budget) / kGiB) * kGiB);
  if (asked > b.budget) b.asked = asked;
  b.budget = std::min(asked, b.budget);
  return b;
}

}  // namespace q27
