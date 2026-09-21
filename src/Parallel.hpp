#pragma once

#include <cstddef>

#include <version>

#if defined(__NVCOMPILER)
#define PALE_PARALLEL_BACKEND_GPU
#elif defined(__GLIBCXX__)  // libstdc++ PSTL -> TBB
#define PALE_PARALLEL_BACKEND_TBB
#elif defined(_LIBCPP_VERSION)  // libc++ PSTL -> libdispatch
#define PALE_PARALLEL_BACKEND_LIBCXX
#else
#define PALE_PARALLEL_BACKEND_UNKNOWN
#endif

#if defined(PALE_PARALLEL) && defined(PALE_PARALLEL_BACKEND_TBB)
#include <mutex>
#include <optional>

#include <oneapi/tbb/global_control.h>
#endif

// =================================================================================================
constexpr auto set_max_threads([[maybe_unused]] size_t max_threads) noexcept -> bool {
#ifndef PALE_PARALLEL
  return max_threads >= 1;
#elifdef PALE_PARALLEL_BACKEND_TBB
  if (max_threads == 0) { return false; }

  static std::mutex mutex{};
  static std::optional<oneapi::tbb::global_control> limit{};

  std::scoped_lock lock(mutex);
  limit.emplace(oneapi::tbb::global_control::max_allowed_parallelism, max_threads);
  return true;
#else
  return false;
#endif
}
