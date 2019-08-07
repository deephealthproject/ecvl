/** @brief Saturate a value (of any type) to the specified type.

Given an input of any type the saturate_cast function provide an
output return value of the specified type applying saturation. When
the input value in greater than the maximum possible value (max) for
the output type, the max value is returned. When the input value in
lower than the minimum possible value (min) for the output type, the
min value is returned.

@param[in] v Input value (of any type).

@return Input value after cast and saturation.
*/
template<DataType ODT, typename IDT>
typename TypeInfo<ODT>::basetype saturate_cast(IDT v) {
    using basetype = typename TypeInfo<ODT>::basetype;

    if (v > std::numeric_limits<basetype>::max()) {
        return std::numeric_limits<basetype>::max();
    }
    if (v < std::numeric_limits<basetype>::min()) {
        return std::numeric_limits<basetype>::min();
    }

    return static_cast<basetype>(v);
}

/** @brief Saturate a value (of any type) to the specified type.

Given an input of any type the saturate_cast function provide an
output return value of the specified type applying saturation. When
the input value in greater than the maximum possible value (max) for
the output type, the max value is returned. When the input value in
lower than the minimum possible value (min) for the output type, the
min value is returned.

@param[in] v Input value (of any type).

@return Input value after cast and saturation.
*/
template<typename ODT, typename IDT>
ODT saturate_cast(const IDT& v) {

    if (v > std::numeric_limits<ODT>::max()) {
        return std::numeric_limits<ODT>::max();
    }
    if (v < std::numeric_limits<ODT>::min()) {
        return std::numeric_limits<ODT>::min();
    }

    return static_cast<ODT>(v);
}

// Template implementation for the in-place Addition between Image(s)
template <DataType DT1, DataType DT2>
struct StructAdd {
    static void _(Image& src1, const Image& src2, bool saturate) {
        using dsttype = typename TypeInfo<DT1>::basetype;
        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteAdd(*is1, *is2));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 + *is2);
            }
        }
    }
};

// Template specialization for the in-place Addition between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarAddImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteAdd(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p + value);
            }
        }
    }
};


// Template implementation for the in-place subtraction between Image(s)
template <DataType DT1, DataType DT2>
struct StructSub {
    static void _(Image& src1, const Image& src2, bool saturate) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteSub(*is1, *is2));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 - *is2);
            }
        }
    }
};

// Template specialization for the in-place subtraction between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarSubImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteSub(value, p));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p - value);
            }
        }
    }
};


// Template implementation for the in-place Multiplication between Image(s)
template <DataType DT1, DataType DT2>
struct StructMul {
    static void _(Image& src1, const Image& src2, bool saturate) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteMul(*is1, *is2));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 * *is2);
            }
        }
    }
};

// Template specialization for the in-place Multiplication between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarMulImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteMul(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p * value);
            }
        }
    }
};

// Template implementation for the in-place division between Image(s)
template <DataType DT1, DataType DT2, typename ET>
struct StructDiv {
    static void _(Image& src1, const Image& src2, bool saturate, ET epsilon) {
        using dsttype = typename TypeInfo<DT1>::basetype;

        View<DT1> vsrc1(src1);
        ConstView<DT2> vsrc2(src2);
        auto is1 = vsrc1.Begin(), es1 = vsrc1.End();
        auto is2 = vsrc2.Begin();
        for (; is1 != es1; ++is1, ++is2) {
            if (saturate) {
                *is1 = saturate_cast<dsttype>(PromoteDiv(*is1, (*is2 + epsilon)));
            }
            else {
                *is1 = static_cast<dsttype>(*is1 / (*is2 + epsilon));
            }
        }
    }
};

// Template specialization for the in-place division between Image and scalar. 
template<DataType DT, typename T>
struct ImageScalarDivImpl {
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteDiv(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p / value);
            }
        }
    }
};