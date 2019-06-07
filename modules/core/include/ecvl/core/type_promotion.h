#ifndef ECVL_TYPE_PROMOTION_H_
#define ECVL_TYPE_PROMOTION_H_

#include <limits>
#include <type_traits>

#include "ecvl/core/datatype.h"

namespace ecvl {

template<typename T, typename U>
struct larger_arithmetic_type {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");
    static_assert(std::is_arithmetic<U>::value, "U must be arithmetic");
    using type = typename std::conditional_t<(std::numeric_limits<T>::digits < std::numeric_limits<U>::digits), U, T>;
};

template<typename T, typename U>
using larger_arithmetic_type_t = typename larger_arithmetic_type<T, U>::type;


template<typename T, typename U>
struct arithmetic_superior_type {
    using type = typename
        std::conditional_t<std::is_floating_point<T>::value && std::is_floating_point<U>::value, larger_arithmetic_type_t<T, U>,
        std::conditional_t<std::is_floating_point<T>::value, T,
        std::conditional_t<std::is_floating_point<U>::value, U,
        larger_arithmetic_type_t<T, U>>>>;
};

template<typename T, typename U>
using arithmetic_superior_type_t = typename arithmetic_superior_type<T, U>::type;


template<typename T, typename U>
struct promote_superior_type {
    using superior_type = arithmetic_superior_type_t<T, U>;

    using type = typename
        std::conditional_t<(sizeof(T) == 8u || sizeof(U) == 8u), double,
        std::conditional_t<std::is_floating_point<superior_type>::value, superior_type,
        std::conditional_t<(std::numeric_limits<superior_type>::digits < std::numeric_limits<std::int16_t>::digits), std::int16_t,
        std::conditional_t<(std::numeric_limits<superior_type>::digits < std::numeric_limits<std::int32_t>::digits), std::int32_t,
        std::conditional_t<(std::numeric_limits<superior_type>::digits < std::numeric_limits<std::int64_t>::digits), std::int64_t, double>>>>>;
};

template<typename T, typename U>
using promote_superior_type_t = typename promote_superior_type<T, U>::type;

template<DataType DT, DataType DU>
using promote_superior_type_dt = promote_superior_type_t<TypeInfo_t<DT>, TypeInfo_t<DU>>;

#define PROMOTE_OPERATION(op_name, op_symbol)                               \
template<typename T, typename U>                                            \
promote_superior_type_t<T, U> Promote ## op_name(T rhs, U lhs) {            \
    using dsttype = promote_superior_type_t<T, U>;                          \
    return static_cast<dsttype>(rhs) op_symbol static_cast<dsttype>(lhs);   \
}                                             

PROMOTE_OPERATION(Add, +)
PROMOTE_OPERATION(Sub, -)
PROMOTE_OPERATION(Mul, *)
PROMOTE_OPERATION(Div, /)

} // namespace ecvl

#endif // ECVL_TYPE_PROMOTION_H_