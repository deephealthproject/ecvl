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

// Template specialization for the in-place Addition between Image and scalar.
template<DataType DT, typename T>
struct ImageScalarAddImpl
{
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

// Template specialization for the in-place subtraction between Image and scalar.
template<DataType DT, typename T>
struct ImageScalarSubImpl
{
    static void _(Image& img, T value, bool saturate)
    {
        View<DT> v(img);
        auto i = v.Begin(), e = v.End();
        for (; i != e; ++i) {
            auto& p = *i;
            if (saturate) {
                p = saturate_cast<DT>(PromoteSub(p, value));
            }
            else {
                p = static_cast<typename TypeInfo<DT>::basetype>(p - value);
            }
        }
    }
};



// Template specialization for the in-place Multiplication between Image and scalar.
template<DataType DT, typename T>
struct ImageScalarMulImpl
{
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

// Template specialization for the in-place division between Image and scalar.
template<DataType DT, typename T>
struct ImageScalarDivImpl
{
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