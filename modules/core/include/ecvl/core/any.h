/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

// We haven't checked which any to include yet
#ifndef INCLUDE_STD_ANY_EXPERIMENTAL

// Check for feature test macro for <any>
#   if defined(__cpp_lib_any)
#       define INCLUDE_STD_ANY_EXPERIMENTAL 0

// Check for feature test macro for <experimental/any>
#   elif defined(__cpp_lib_experimental_any)
#       define INCLUDE_STD_ANY_EXPERIMENTAL 1

// We can't check if headers exist...
// Let's assume experimental to be safe
#   elif !defined(__has_include)
#       define INCLUDE_STD_ANY_EXPERIMENTAL 1

// Check if the header "<any>" exists
#   elif __has_include(<any>)

// If we're compiling on Visual Studio and are not compiling with C++17, we need to use experimental
#       ifdef _MSC_VER

// Check and include header that defines "_HAS_CXX17"
#           if __has_include(<yvals_core.h>)
#               include <yvals_core.h>

// Check for enabled C++17 support
#               if defined(_HAS_CXX17) && _HAS_CXX17
// We're using C++17, so let's use the normal version
#                   define INCLUDE_STD_ANY_EXPERIMENTAL 0
#               endif
#           endif

// If the macro isn't defined yet, that means any of the other VS specific checks failed, so we need to use experimental
#           ifndef INCLUDE_STD_ANY_EXPERIMENTAL
#               define INCLUDE_STD_ANY_EXPERIMENTAL 1
#           endif

// Not on Visual Studio. Let's use the normal version
#       else // #ifdef _MSC_VER
#           define INCLUDE_STD_ANY_EXPERIMENTAL 0
#       endif

// Check if the header "<any>" exists
#   elif __has_include(<experimental/any>)
#       define INCLUDE_STD_ANY_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#   else
#       error Could not find system header "<any>" or "<experimental/any>"
#   endif

// We previously determined that we need the experimental version
#   if INCLUDE_STD_ANY_EXPERIMENTAL
#       include <experimental/any>
namespace ecvl
{
using any = std::experimental::any;
}
#   else
#       include <any>
namespace ecvl
{
using any = std::any;
}
#   endif

#endif // #ifndef INCLUDE_STD_ANY_EXPERIMENTAL