/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

// We haven't checked which optional to include yet
#ifndef INCLUDE_STD_OPTIONAL_EXPERIMENTAL

// Check for feature test macro for <optional>
#   if defined(__cpp_lib_optional)
#       define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 0

// Check for feature test macro for <experimental/optional>
#   elif defined(__cpp_lib_experimental_optional)
#       define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 1

// We can't check if headers exist...
// Let's assume experimental to be safe
#   elif !defined(__has_include)
#       define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 1

// Check if the header "<optional>" exists
#   elif __has_include(<optional>)

// If we're compiling on Visual Studio and are not compiling with C++17, we need to use experimental
#       ifdef _MSC_VER

// Check and include header that defines "_HAS_CXX17"
#           if __has_include(<yvals_core.h>)
#               include <yvals_core.h>

// Check for enabled C++17 support
#               if defined(_HAS_CXX17) && _HAS_CXX17
// We're using C++17, so let's use the normal version
#                   define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 0
#               endif
#           endif

// If the macro isn't defined yet, that means any of the other VS specific checks failed, so we need to use experimental
#           ifndef INCLUDE_STD_OPTIONAL_EXPERIMENTAL
#               define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 1
#           endif

// Not on Visual Studio. Let's use the normal version
#       else // #ifdef _MSC_VER
#           define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 0
#       endif

// Check if the header "<optional>" exists
#   elif __has_include(<experimental/optional>)
#       define INCLUDE_STD_OPTIONAL_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#   else
#       error Could not find system header "<optional>" or "<experimental/optional>"
#   endif

// We previously determined that we need the experimental version
#   if INCLUDE_STD_OPTIONAL_EXPERIMENTAL
#       include <experimental/optional>
namespace ecvl
{
template<typename T>
using optional = std::experimental::optional<T>;

using bad_optional_access = std::experimental::bad_optional_access;
static auto& nullopt = std::experimental::nullopt;
}
#   else
#       include <optional>
namespace ecvl
{
template<typename T>
using optional = std::optional<T>;

using bad_optional_access = std::bad_optional_access;
static auto& nullopt = std::nullopt;
}
#   endif

#endif // #ifndef INCLUDE_STD_OPTIONAL_EXPERIMENTAL