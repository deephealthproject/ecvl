#include "ecvl/core/arithmetic.h"

namespace ecvl {

#define STANDARD_INPLACE_OPERATION(Function, TemplateImplementation) \
void Function(Image& src1_dst, const Image& src2)                    \
{                                                                    \
    static constexpr Table2D<TemplateImplementation> table;          \
    table(src1_dst.elemtype_, src2.elemtype_)(src1_dst, src2);       \
}
STANDARD_INPLACE_OPERATION(Add, StructAdd)
STANDARD_INPLACE_OPERATION(Sub, StructSub)
STANDARD_INPLACE_OPERATION(Mul, StructMul)
STANDARD_INPLACE_OPERATION(Div, StructDiv)

} // namespace ecvl