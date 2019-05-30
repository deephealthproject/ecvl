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

Image& Mul(Image& img, double d, bool saturate) {

    if (img.contiguous_) {
        switch (img.elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Mul<ContiguousView<DataType::name>>(img, d, saturate);
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("How did you get here?");
        }
    }
    else {
        switch (img.elemtype_)
        {
#define ECVL_TUPLE(name, ...) case DataType::name: return Mul<View<DataType::name>>(img, d, saturate);
#include "ecvl/core/datatype_existing_tuples.inc.h"
#undef ECVL_TUPLE
        default:
            throw std::runtime_error("How did you get here?");
        }
    }
}

//Image& Sum(Image& img, int v, bool saturate) {
//
//    if (img.contiguous_) {
//        switch (img.elemtype_)
//        {
//#define ECVL_TUPLE(name, ...) case DataType::name: return Sum<ContiguousView<DataType::name>>(img, v, saturate);
//#include "ecvl/core/datatype_existing_tuples.inc"
//#undef ECVL_TUPLE
//        default:
//            throw std::runtime_error("How did you get here?");
//        }
//    }
//    else {
//        switch (img.elemtype_)
//        {
//#define ECVL_TUPLE(name, ...) case DataType::name: return Sum<View<DataType::name>>(img, v, saturate);
//#include "ecvl/core/datatype_existing_tuples.inc"
//#undef ECVL_TUPLE
//        default:
//            throw std::runtime_error("How did you get here?");
//        }
//    }
//}


} // namespace ecvl