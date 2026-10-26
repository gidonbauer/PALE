#pragma once

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#ifdef PALE_PROFILE

#include <cstdlib>

namespace {
struct CaliperDefaultConfig {
  CaliperDefaultConfig() { setenv("CALI_CONFIG", "runtime-report", 0); }
} caliper_default_config;
}  // namespace

#include <caliper/cali.h>

#define PALE_PROFILE_FUNCTION CALI_CXX_MARK_FUNCTION
#define PALE_PROFILE_SCOPE(name) CALI_CXX_MARK_SCOPE(name)
#define PALE_PROFILE_BEGIN(name) CALI_MARK_BEGIN(name)
#define PALE_PROFILE_END(name) CALI_MARK_END(name)

#else

#define PALE_PROFILE_FUNCTION (void)0
#define PALE_PROFILE_SCOPE(name) (void)0
#define PALE_PROFILE_BEGIN(name) (void)0
#define PALE_PROFILE_END(name) (void)0

#endif

// NOLINTEND(cppcoreguidelines-macro-usage)
