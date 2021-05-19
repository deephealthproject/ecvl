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

#ifndef AUGMENTATIONS_H_
#define AUGMENTATIONS_H_

#include "ecvl/core/arithmetic.h"
#include "ecvl/core/imgproc.h"
#include <array>
#include <map>
#include <memory>
#include <random>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace ecvl
{
#define ECVL_ERROR_AUGMENTATION_NAME throw std::runtime_error(ECVL_ERROR_MSG "Cannot load augmentation name");
#define ECVL_ERROR_AUGMENTATION_FORMAT throw std::runtime_error(ECVL_ERROR_MSG "Format error while loading augmentation parameters");

class param_list;

class param
{
    static std::istream& read_until(std::istream& is, std::string& s, const std::string& list)
    {
        s.clear();
        while (is.peek() && is && list.find(is.peek()) == list.npos) {
            s += is.get();
        }
        return is;
    }

    void read_vals(std::istream& is, char closing_char)
    {
        double val;
        char next_char;
        do {
            is >> val;
            if (!is) {
                break;
            }
            vals_.push_back(val);
            is >> next_char;
        } while (next_char == ',');
        if (!is || next_char != closing_char) {
            std::cerr << "Error while reading values of parameter " << name_ << "\n"; // TODO: standardize
            throw std::runtime_error("Cannot read parameter value"); // TODO: standardize
        }
    }

public:
    enum class type { range, vector, number, string };

    static const char* to_string(type t)
    {
        switch (t) {
        case type::range: return "range";
        case type::vector: return "vector";
        case type::number: return "number";
        case type::string: return "string";
        default:
            ECVL_ERROR_NOT_REACHABLE_CODE
        }
    }

    std::string name_;
    type type_;
    std::vector<double> vals_;
    std::string str_;

    param() {}
    param(std::istream& is)
    {
        is >> std::ws;
        read_until(is, name_, " =");
        char next_char;
        is >> std::ws >> next_char;
        if (next_char != '=') {
            throw std::runtime_error("Cannot read parameter name"); // TODO: standardize
        }
        is >> std::ws;
        next_char = is.peek();
        if (next_char == '[') { // range
            is.ignore();
            type_ = type::range;
            read_vals(is, ']');
        } else if (next_char == '(') { // vector
            is.ignore();
            type_ = type::vector;
            read_vals(is, ')');
        } else if (next_char == '"') { // string
            is.ignore();
            type_ = type::string;
            std::getline(is, str_, '"');
        } else {
            type_ = type::number;
            vals_.resize(1);
            is >> vals_[0];
        }
        if (!is) {
            std::cerr << "Error while reading value of parameter " << name_ << "\n"; // TODO: standardize
            throw std::runtime_error("Cannot read parameter value"); // TODO: standardize
        }
    }

    friend class param_list;
    static param_list read(std::istream& is, std::string fn_name_);
};

class param_list
{
    std::unordered_map<std::string, param> m_;
    const std::string fn_name_;
public:
    param_list(std::string fn_name) : fn_name_(move(fn_name)) {}

    auto& operator[](const std::string& s)
    {
        return m_[s];
    }

    bool Get(const std::string& name, param::type type, bool required, param& value)
    {
        auto it = m_.find(name);
        if (it != end(m_)) {
            auto& p = it->second;
            if (p.type_ != type) {
                throw std::runtime_error(fn_name_ + ": " + name + " parameter must be a " + param::to_string(type));
            }
            value = p;
            return true;
        }
        if (required) {
            throw std::runtime_error(fn_name_ + ": " + name + " is a required parameter");
        }
        return false;
    }

    bool GenericGet(const std::string& name, bool required, param& value)
    {
        auto it = m_.find(name);
        if (it != end(m_)) {
            auto& p = it->second;
            value = p;
            return true;
        }
        if (required) {
            throw std::runtime_error(fn_name_ + ": " + name + " is a required parameter");
        }
        return false;
    }
};

/** @brief Augmentations parameters.

This class represent the augmentations parameters which must be randomly generated in a specific range.

@anchor AugmentationParam
*/
class AugmentationParam
{
public:
    double min_, max_, value_;

    AugmentationParam() = default;
    AugmentationParam(const double min, const double max) : min_(min), max_(max) {}

    /** @brief Generate the random value between min_ and max_.
    */
    void GenerateValue()
    {
        value_ = std::uniform_real_distribution<>(min_, max_)(re_);
    }
    static std::default_random_engine re_;

    static constexpr unsigned seed_min = std::numeric_limits<unsigned>::min();
    static constexpr unsigned seed_max = std::numeric_limits<unsigned>::max();

    /** @brief Set a fixed seed for the random generated values. Useful to reproduce experiments with same augmentations.
    @param[in] seed Value of the seed for the random engine.
    */
    static void SetSeed(unsigned seed)
    {
        re_.seed(seed);
    }
};

/** @brief Abstract class which represent a generic Augmentation function.

@anchor Augmentation
*/
class Augmentation
{
public:
    std::unordered_map<std::string, AugmentationParam> params_;

    /** @brief Generate the random value for each parameter and call the specialized augmentation functions.
    @param[in,out] img Image on which apply the augmentations.
    @param[in,out] gt Ground truth image on which apply the augmentations.
    */
    void Apply(ecvl::Image& img, const ecvl::Image& gt = Image())
    {
        for (auto& x : params_) {
            x.second.GenerateValue();
        }
        RealApply(img, gt);
    }
    virtual ~Augmentation() = default;

private:
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) = 0;
};

struct AugmentationFactory
{
    static std::shared_ptr<Augmentation> create(std::istream& is)
    {
        std::string name;
        is >> name;
        if (!is) {
            ECVL_ERROR_AUGMENTATION_NAME
        }
        return create(name, is);
    }

    static std::shared_ptr<Augmentation> create(const std::string& name, std::istream& is);
};

/** @brief SequentialAugmentationContainer.

This class represents a container for multiple augmentations which will be sequentially applied to the Dataset images.

@anchor SequentialAugmentationContainer
*/
class SequentialAugmentationContainer : public Augmentation
{
    /** @brief Call the specialized augmentation functions.

    @param[in] img Image on which apply the augmentations.
    @param[in] gt Ground truth image on which apply the augmentations.
    */
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        for (auto& x : augs_) {
            x->Apply(img, gt);
        }
    }
    std::vector<std::shared_ptr<Augmentation>> augs_;   /**< @brief vector containing the Augmentation to be applied */
public:
    template<typename ...Ts>
    SequentialAugmentationContainer(Ts&&... t) : augs_({ std::make_shared<Ts>(std::forward<Ts>(t))... }) {}

    SequentialAugmentationContainer(std::vector<std::shared_ptr<Augmentation>> augs) : augs_(augs) {}

    SequentialAugmentationContainer(std::istream& is)
    {
        while (true) {
            std::string name;
            is >> name;
            if (!is) {
                ECVL_ERROR_AUGMENTATION_NAME
            }
            if (name == "end") {
                break;
            }
            augs_.emplace_back(AugmentationFactory::create(name, is));
        }
    }
};

/** @brief OneOfAugmentationContainer.

This class represents a container for multiple augmentations from which one will be randomly chosen.
The chosen augmentation will be applied with a probability that must be specified by the user.

@anchor OneOfAugmentationContainer
*/
class OneOfAugmentationContainer : public Augmentation
{
    /** @brief Call the specialized augmentation functions.

    @param[in] img Image on which apply the augmentations.
    @param[in] gt Ground truth image on which apply the augmentations.
    */
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        int index = std::uniform_int_distribution<>(0, vsize(augs_) - 1)(AugmentationParam::re_);
        if (params_["p"].value_ <= p_) {
            augs_[index]->Apply(img, gt);
        }
    }
    std::vector<std::shared_ptr<Augmentation>> augs_;   /**< @brief vector containing the Augmentation to be applied */
    double p_;
public:
    template<typename ...Ts>
    OneOfAugmentationContainer(double p, Ts&&... t) : p_(p), augs_({ std::make_shared<Ts>(std::forward<Ts>(t))... })
    {
        params_["p"] = AugmentationParam(0, 1);
    }

    OneOfAugmentationContainer(double p, std::vector<std::shared_ptr<Augmentation>> augs) : p_(p), augs_(augs)
    {
        params_["p"] = AugmentationParam(0, 1);
    }

    OneOfAugmentationContainer(std::istream& is)
    {
        param p;
        try {
            auto m = param::read(is, "OneOfAugmentationContainer");
            if (m.Get("p", param::type::number, true, p)) {
                p_ = p.vals_[0];
            }
        } catch (std::runtime_error&) {
            std::cout << ECVL_ERROR_MSG "The first parameter in OneOfAugmentationContainer must be the probability p" << std::endl;
            ECVL_ERROR_AUGMENTATION_FORMAT
        }

        while (true) {
            std::string name;
            is >> name;
            if (!is) {
                ECVL_ERROR_AUGMENTATION_NAME
            }
            if (name == "end") {
                break;
            }
            augs_.emplace_back(AugmentationFactory::create(name, is));
        }
    }
};

InterpolationType StrToInterpolationType(const std::string& interp, const std::string& aug_name);

///////////////////////////////////////////////////////////////////////////////////
// Augmentations
///////////////////////////////////////////////////////////////////////////////////

/** @brief Augmentation wrapper for ecvl::Rotate2D.

@anchor AugRotate
*/
class AugRotate : public Augmentation
{
    std::vector<double> center_;
    double scale_;
    InterpolationType interp_, gt_interp_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto angle = params_["angle"].value_;
        Rotate2D(img, img, angle, center_, scale_, interp_);
        if (!gt.IsEmpty()) {
            Rotate2D(gt, const_cast<Image&>(gt), angle, center_, scale_, gt_interp_);
        }
    }
public:
    /** @brief AugRotate constructor

    @param[in] angle Parameter which determines the range of degrees [min,max] to randomly select from.
    @param[in] center A std::vector<double> representing the coordinates of the rotation center.
                      If empty, the center of the image is used.
    @param[in] scale Optional scaling factor.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    @param[in] gt_interp InterpolationType to be used for ground truth. Default is InterpolationType::nearest.
    */
    AugRotate(const std::array<double, 2>& angle,
        const std::vector<double>& center = {},
        const double& scale = 1.,
        const InterpolationType& interp = InterpolationType::linear,
        const InterpolationType& gt_interp = InterpolationType::nearest)
        : center_(center), scale_(scale), interp_(interp), gt_interp_(gt_interp)
    {
        params_["angle"] = AugmentationParam(angle[0], angle[1]);
    }

    AugRotate(std::istream& is)
    {
        auto m = param::read(is, "AugRotate");
        param p;

        m.Get("angle", param::type::range, true, p);
        params_["angle"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        if (m.Get("center", param::type::vector, false, p)) {
            center_ = p.vals_;
        }

        scale_ = 1.;
        if (m.Get("scale", param::type::number, false, p)) {
            scale_ = p.vals_[0];
        }

        interp_ = InterpolationType::linear;
        gt_interp_ = InterpolationType::nearest;

        if (m.Get("interp", param::type::string, false, p)) {
            interp_ = StrToInterpolationType(p.str_, "AugRotate");
        }
        if (m.Get("gt_interp", param::type::string, false, p)) {
            gt_interp_ = StrToInterpolationType(p.str_, "AugRotate");
        }
    }
};

/** @brief Augmentation wrapper for ecvl::ResizeDim.

@anchor AugResizeDim
*/
class AugResizeDim : public Augmentation
{
    std::vector<int> dims_;
    InterpolationType interp_, gt_interp_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ResizeDim(img, img, dims_, interp_);
        if (!gt.IsEmpty()) {
            ResizeDim(gt, const_cast<Image&>(gt), dims_, gt_interp_);
        }
    }
public:
    /** @brief AugResizeDim constructor

    @param[in] dims std::vector<int> that specifies the new size of each dimension.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    @param[in] gt_interp InterpolationType to be used for ground truth. Default is InterpolationType::nearest.
    */
    AugResizeDim(const std::vector<int>& dims,
        const InterpolationType& interp = InterpolationType::linear,
        const InterpolationType& gt_interp = InterpolationType::nearest)
        : dims_{ dims }, interp_(interp), gt_interp_(gt_interp) {}

    AugResizeDim(std::istream& is)
    {
        auto m = param::read(is, "AugResizeDim");
        param p;

        m.Get("dims", param::type::vector, true, p);
        for (const auto& x : p.vals_) {
            dims_.emplace_back(static_cast<int>(x));
        }

        interp_ = InterpolationType::linear;
        gt_interp_ = InterpolationType::nearest;

        if (m.Get("interp", param::type::string, false, p)) {
            interp_ = StrToInterpolationType(p.str_, "");
        }
        if (m.Get("gt_interp", param::type::string, false, p)) {
            gt_interp_ = StrToInterpolationType(p.str_, "AugResizeDim");
        }
    }
};

/** @brief Augmentation wrapper for ecvl::ResizeScale.

@anchor AugResizeScale
*/
class AugResizeScale : public Augmentation
{
    std::vector<double> scale_;
    InterpolationType interp_, gt_interp_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ResizeScale(img, img, scale_, interp_);
        if (!gt.IsEmpty()) {
            ResizeScale(gt, const_cast<Image&>(gt), scale_, gt_interp_);
        }
    }
public:
    /** @brief AugResizeScale constructor

    @param[in] scale std::vector<double> that specifies the scale to apply to each dimension.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    @param[in] gt_interp InterpolationType to be used for ground truth. Default is InterpolationType::nearest.
    */
    AugResizeScale(const std::vector<double>& scale,
        const InterpolationType& interp = InterpolationType::linear,
        const InterpolationType& gt_interp = InterpolationType::nearest
    ) : scale_{ scale }, interp_(interp), gt_interp_(gt_interp){}

    AugResizeScale(std::istream& is)
    {
        auto m = param::read(is, "AugResizeScale");
        param p;

        m.Get("scale", param::type::vector, true, p);
        scale_ = p.vals_;

        interp_ = InterpolationType::linear;
        gt_interp_ = InterpolationType::nearest;

        if (m.Get("interp", param::type::string, false, p)) {
            interp_ = StrToInterpolationType(p.str_, "AugResizeScale");
        }
        if (m.Get("gt_interp", param::type::string, false, p)) {
            gt_interp_ = StrToInterpolationType(p.str_, "AugResizeScale");
        }
    }
};

/** @brief Augmentation wrapper for ecvl::Flip2D.

@anchor AugFlip
*/
class AugFlip : public Augmentation
{
    double p_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        if (p <= p_) {
            Flip2D(img, img);
            if (!gt.IsEmpty()) {
                Flip2D(gt, const_cast<Image&>(gt));
            }
        }
    }
public:
    /** @brief AugFlip constructor

    @param[in] p Probability of each image to get flipped.
    */
    AugFlip(double p = 0.5) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }

    AugFlip(std::istream& is) : AugFlip()
    {
        auto m = param::read(is, "AugFlip");
        param p;

        if (m.Get("p", param::type::number, false, p)) {
            p_ = p.vals_[0];
        }
    }
};

/** @brief Augmentation wrapper for ecvl::Mirror2D.

@anchor AugMirror
*/
class AugMirror : public Augmentation
{
    double p_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        if (p <= p_) {
            Mirror2D(img, img);
            if (!gt.IsEmpty()) {
                Mirror2D(gt, const_cast<Image&>(gt));
            }
        }
    }
public:
    /** @brief AugMirror constructor

    @param[in] p Probability of each image to get mirrored.
    */
    AugMirror(double p = 0.5) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }

    AugMirror(std::istream& is) : AugMirror()
    {
        auto m = param::read(is, "AugMirror");
        param p;

        if (m.Get("p", param::type::number, false, p)) {
            p_ = p.vals_[0];
        }
    }
};

/** @brief Augmentation wrapper for ecvl::GaussianBlur.

@anchor AugGaussianBlur
*/
class AugGaussianBlur : public Augmentation
{
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto sigma = params_["sigma"].value_;
        GaussianBlur(img, img, sigma);
    }
public:
    /** @brief AugGaussianBlur constructor

    @param[in] sigma Parameter which determines the range of sigma [min,max] to randomly select from.
    */
    AugGaussianBlur(const std::array<double, 2>& sigma)
    {
        params_["sigma"] = AugmentationParam(sigma[0], sigma[1]);
    }

    AugGaussianBlur(std::istream& is)
    {
        auto m = param::read(is, "AugGaussianBlur");
        param p;

        m.Get("sigma", param::type::range, true, p);
        params_["sigma"] = AugmentationParam(p.vals_[0], p.vals_[1]);
    }
};

/** @brief Augmentation wrapper for ecvl::AdditiveLaplaceNoise.

@anchor AugAdditiveLaplaceNoise
*/
class AugAdditiveLaplaceNoise : public Augmentation
{
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto std_dev = params_["std_dev"].value_;
        AdditiveLaplaceNoise(img, img, std_dev);
    }
public:
    /** @brief AugAdditiveLaplaceNoise constructor

    @param[in] std_dev Parameter which determines the range of values [min,max] to randomly select the standard deviation of the noise generating distribution.
                       Suggested values are around 255 * 0.05 for uint8 Images.
    */
    AugAdditiveLaplaceNoise(const std::array<double, 2>& std_dev)
    {
        params_["std_dev"] = AugmentationParam(std_dev[0], std_dev[1]);
    }

    AugAdditiveLaplaceNoise(std::istream& is)
    {
        auto m = param::read(is, "AugAdditiveLaplaceNoise");
        param p;

        m.Get("std_dev", param::type::range, true, p);
        params_["std_dev"] = AugmentationParam(p.vals_[0], p.vals_[1]);
    }
};

/** @brief Augmentation wrapper for ecvl::AdditivePoissonNoise.

@anchor AugAdditivePoissonNoise
*/
class AugAdditivePoissonNoise : public Augmentation
{
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto lambda = params_["lambda"].value_;
        AdditivePoissonNoise(img, img, lambda);
    }
public:
    /** @brief AugAdditivePoissonNoise constructor

    @param[in] lambda Parameter which determines the range of values [min,max] to randomly select the lambda of the noise generating distribution.
                      Suggested values are around 0.0 to 10.0.
    */
    AugAdditivePoissonNoise(const std::array<double, 2>& lambda)
    {
        params_["lambda"] = AugmentationParam(lambda[0], lambda[1]);
    }

    AugAdditivePoissonNoise(std::istream& is)
    {
        auto m = param::read(is, "AugAdditivePoissonNoise");
        param p;

        m.Get("lambda", param::type::range, true, p);
        params_["lambda"] = AugmentationParam(p.vals_[0], p.vals_[1]);
    }
};

/** @brief Augmentation wrapper for ecvl::GammaContrast.

@anchor AugGammaContrast
*/
class AugGammaContrast : public Augmentation
{
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto gamma = params_["gamma"].value_;
        GammaContrast(img, img, gamma);
    }
public:
    /** @brief AugGammaContrast constructor

    @param[in] gamma Parameter which determines the range of values [min,max] to randomly select the exponent for the contrast adjustment.
                     Suggested values are around 0.5 to 2.0.
    */
    AugGammaContrast(const std::array<double, 2>& gamma)
    {
        params_["gamma"] = AugmentationParam(gamma[0], gamma[1]);
    }

    AugGammaContrast(std::istream& is)
    {
        auto m = param::read(is, "AugGammaContrast");
        param p;

        m.Get("gamma", param::type::range, true, p);
        params_["gamma"] = AugmentationParam(p.vals_[0], p.vals_[1]);
    }
};

/** @brief Augmentation wrapper for ecvl::CoarseDropout.

@anchor AugCoarseDropout
*/
class AugCoarseDropout : public Augmentation
{
    double per_channel_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        const auto drop_size = params_["drop_size"].value_;
        const bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        CoarseDropout(img, img, p, drop_size, per_channel);
    }
public:
    /** @brief AugCoarseDropout constructor

    @param[in] p Parameter which determines the range of values [min,max] to randomly select the probability of any rectangle being set to zero.
    @param[in] drop_size Parameter which determines the range of values [min,max] to randomly select the size of rectangles in percentage of the input Image.
    @param[in] per_channel Probability of each image to use the same value for all channels of a pixel or not.
    */
    AugCoarseDropout(const std::array<double, 2>& p, const std::array<double, 2>& drop_size, const double& per_channel) : per_channel_(per_channel)
    {
        assert(per_channel >= 0 && per_channel <= 1);
        params_["p"] = AugmentationParam(p[0], p[1]);
        params_["drop_size"] = AugmentationParam(drop_size[0], drop_size[1]);
        params_["per_channel"] = AugmentationParam(0, 1);
    }
    AugCoarseDropout(std::istream& is)
    {
        auto m = param::read(is, "AugCoarseDropout");
        param p;

        m.Get("p", param::type::range, true, p);
        params_["p"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        m.Get("drop_size", param::type::range, true, p);
        params_["drop_size"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        m.Get("per_channel", param::type::number, true, p);
        params_["per_channel"] = AugmentationParam(0, 1);
        per_channel_ = p.vals_[0];
    }
};

/** @brief Augmentation wrapper for ecvl::Transpose.

@anchor AugTranspose
*/
class AugTranspose : public Augmentation
{
    double p_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        if (p <= p_) {
            Transpose(img, img);
            if (!gt.IsEmpty()) {
                Transpose(gt, const_cast<Image&>(gt));
            }
        }
    }
public:
    /** @brief AugTranspose constructor

    @param[in] p Probability of each image to get transposed.
    */
    AugTranspose(double p = 0.5) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }

    AugTranspose(std::istream& is) : AugTranspose()
    {
        auto m = param::read(is, "AugTranspose");
        param p;

        if (m.Get("p", param::type::number, false, p)) {
            p_ = p.vals_[0];
        }
    }
};

/** @brief Augmentation wrapper for brightness adjustment.

@anchor AugBrightness
*/
class AugBrightness : public Augmentation
{
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto beta = params_["beta"].value_;
        Add(img, beta, img);
    }
public:
    /** @brief AugBrightness constructor

    @param[in] beta Parameter which determines the range of values [min,max] to randomly select the value for the brightness adjustment.
                    Suggested values are around 0 to 100.
    */
    AugBrightness(const std::array<double, 2>& beta)
    {
        params_["beta"] = AugmentationParam(beta[0], beta[1]);
    }

    AugBrightness(std::istream& is)
    {
        auto m = param::read(is, "AugBrightness");

        param p;

        m.Get("beta", param::type::range, true, p);
        params_["beta"] = AugmentationParam(p.vals_[0], p.vals_[1]);
    }
};

/** @brief Augmentation wrapper for ecvl::GridDistortion.

@anchor AugGridDistortion
*/
class AugGridDistortion : public Augmentation
{
    std::array<float, 2> distort_limit_;
    InterpolationType interp_;
    BorderType border_type_;
    int border_value_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto num_steps = params_["num_steps"].value_;
        const auto seed = params_["seed"].value_;
        GridDistortion(img, img, static_cast<int>(num_steps), distort_limit_, interp_, border_type_,
            border_value_, static_cast<unsigned>(seed));
        if (!gt.IsEmpty()) {
            GridDistortion(gt, const_cast<Image&>(gt), static_cast<int>(num_steps), distort_limit_,
                interp_, border_type_, border_value_, static_cast<unsigned>(seed));
        }
    }
public:
    /** @brief AugGridDistortion constructor

    @param[in] num_steps Parameter which determines the range of values [min,max] to randomly select the number of grid cells on each side.
    @param[in] distort_limit Parameter which determines the range of values [min,max] to randomly select the distortion steps.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    @param[in] border_type Flag used to specify the pixel extrapolation method. Default is BorderType::BORDER_REFLECT_101.
    @param[in] border_value Integer padding value if border_type is BorderType::BORDER_CONSTANT.
    */
    AugGridDistortion(const std::array<int, 2>& num_steps,
        const std::array<float, 2>& distort_limit,
        const InterpolationType& interp = InterpolationType::linear,
        const BorderType& border_type = BorderType::BORDER_REFLECT_101,
        const int& border_value = 0)
        : distort_limit_(distort_limit), interp_(interp), border_type_(border_type), border_value_(border_value)
    {
        params_["num_steps"] = AugmentationParam(num_steps[0], num_steps[1]);
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);
    }

    AugGridDistortion(std::istream& is)
    {
        auto m = param::read(is, "AugGridDistortion");

        param p;

        m.Get("num_steps", param::type::range, true, p);
        params_["num_steps"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        // seed is managed by AugmentationParam
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);

        m.Get("distort_limit", param::type::range, true, p);
        distort_limit_ = { static_cast<float>(p.vals_[0]), static_cast<float>(p.vals_[1]) };

        interp_ = InterpolationType::linear;
        if (m.Get("interp", param::type::string, false, p)) {
            interp_ = StrToInterpolationType(p.str_, "AugGridDistortion");
        }

        border_type_ = BorderType::BORDER_REFLECT_101;
        if (m.Get("border_type", param::type::string, false, p)) {
            if (p.str_ == "constant") {
                border_type_ = BorderType::BORDER_CONSTANT;
            } else if (p.str_ == "replicate") {
                border_type_ = BorderType::BORDER_REPLICATE;
            } else if (p.str_ == "reflect") {
                border_type_ = BorderType::BORDER_REFLECT;
            } else if (p.str_ == "wrap") {
                border_type_ = BorderType::BORDER_WRAP;
            } else if (p.str_ == "reflect_101") {
                border_type_ = BorderType::BORDER_REFLECT_101;
            } else if (p.str_ == "transparent") {
                border_type_ = BorderType::BORDER_TRANSPARENT;
            } else {
                throw std::runtime_error("AugGridDistortion: invalid border type"); // TODO: standardize
            }
        }

        border_value_ = 0;
        m.Get("border_value", param::type::number, false, p);
        border_value_ = static_cast<int>(p.vals_[0]);
    }
};

/** @brief Augmentation wrapper for ecvl::ElasticTransform.

@anchor AugElasticTransform
*/
class AugElasticTransform : public Augmentation
{
    InterpolationType interp_;
    BorderType border_type_;
    int border_value_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto alpha = params_["alpha"].value_;
        const auto sigma = params_["sigma"].value_;
        const auto seed = params_["seed"].value_;
        ElasticTransform(img, img, alpha, sigma, interp_, border_type_, border_value_, static_cast<unsigned>(seed));
        if (!gt.IsEmpty()) {
            ElasticTransform(gt, const_cast<Image&>(gt), alpha, sigma, interp_,
                border_type_, border_value_, static_cast<unsigned>(seed));
        }
    }
public:
    /** @brief AugElasticTransform constructor

    @param[in] alpha Parameter which determines the range of values [min,max] to randomly select the scaling factor that controls the intensity of the deformation.
    @param[in] sigma Parameter which determines the range of values [min,max] to randomly select the gaussian kernel standard deviation.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    @param[in] border_type Flag used to specify the pixel extrapolation method. Default is BorderType::BORDER_REFLECT_101.
    @param[in] border_value Integer padding value if border_type is BorderType::BORDER_CONSTANT.
    */
    AugElasticTransform(const std::array<double, 2>& alpha,
        const std::array<double, 2>& sigma,
        const InterpolationType& interp = InterpolationType::linear,
        const BorderType& border_type = BorderType::BORDER_REFLECT_101,
        const int& border_value = 0)
        : interp_(interp), border_type_(border_type), border_value_(border_value)
    {
        params_["alpha"] = AugmentationParam(alpha[0], alpha[1]);
        params_["sigma"] = AugmentationParam(sigma[0], sigma[1]);
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);
    }

    AugElasticTransform(std::istream& is)
    {
        auto m = param::read(is, "AugElasticTransform");

        param p;

        m.Get("alpha", param::type::range, true, p);
        params_["alpha"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        m.Get("sigma", param::type::range, true, p);
        params_["sigma"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        // seed is managed by AugmentationParam
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);

        interp_ = InterpolationType::linear;
        if (m.Get("interp", param::type::string, false, p)) {
            interp_ = StrToInterpolationType(p.str_, "AugElasticTransform");
        }

        border_type_ = BorderType::BORDER_REFLECT_101;
        if (m.Get("border_type", param::type::string, false, p)) {
            if (p.str_ == "constant") {
                border_type_ = BorderType::BORDER_CONSTANT;
            } else if (p.str_ == "replicate") {
                border_type_ = BorderType::BORDER_REPLICATE;
            } else if (p.str_ == "reflect") {
                border_type_ = BorderType::BORDER_REFLECT;
            } else if (p.str_ == "wrap") {
                border_type_ = BorderType::BORDER_WRAP;
            } else if (p.str_ == "reflect_101") {
                border_type_ = BorderType::BORDER_REFLECT_101;
            } else if (p.str_ == "transparent") {
                border_type_ = BorderType::BORDER_TRANSPARENT;
            } else {
                throw std::runtime_error("AugGridDistortion: invalid border type"); // TODO: standardize
            }
        }

        border_value_ = 0;
        m.Get("border_value", param::type::number, false, p);
        border_value_ = static_cast<int>(p.vals_[0]);
    }
};

/** @brief Augmentation wrapper for ecvl::OpticalDistortion.

@anchor AugOpticalDistortion
*/
class AugOpticalDistortion : public Augmentation
{
    std::array<float, 2> distort_limit_;
    std::array<float, 2> shift_limit_;
    InterpolationType interp_;
    BorderType border_type_;
    int border_value_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto seed = params_["seed"].value_;
        OpticalDistortion(img, img, distort_limit_, shift_limit_, interp_, border_type_,
            border_value_, static_cast<unsigned>(seed));
        if (!gt.IsEmpty()) {
            OpticalDistortion(gt, const_cast<Image&>(gt), distort_limit_, shift_limit_, interp_, border_type_,
                border_value_, static_cast<unsigned>(seed));
        }
    }
public:
    /** @brief AugOpticalDistortion constructor

    @param[in] distort_limit Parameter which determines the range of values [min,max] to randomly select the distortion steps.
    @param[in] shift_limit Parameter which determines the range of values [min,max] to randomly select the image shifting.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    @param[in] border_type Flag used to specify the pixel extrapolation method. Default is BorderType::BORDER_REFLECT_101.
    @param[in] border_value Integer padding value if border_type is BorderType::BORDER_CONSTANT.
    */
    AugOpticalDistortion(const std::array<float, 2>& distort_limit,
        const std::array<float, 2>& shift_limit,
        const InterpolationType& interp = InterpolationType::linear,
        const BorderType& border_type = BorderType::BORDER_REFLECT_101,
        const int& border_value = 0)
        : distort_limit_(distort_limit), shift_limit_(shift_limit), interp_(interp), border_type_(border_type), border_value_(border_value)
    {
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);
    }

    AugOpticalDistortion(std::istream& is)
    {
        auto m = param::read(is, "AugOpticalDistortion");

        param p;

        // seed is managed by AugmentationParam
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);

        m.Get("distort_limit", param::type::range, true, p);
        distort_limit_ = { static_cast<float>(p.vals_[0]), static_cast<float>(p.vals_[1]) };

        m.Get("shift_limit", param::type::range, true, p);
        shift_limit_ = { static_cast<float>(p.vals_[0]), static_cast<float>(p.vals_[1]) };

        interp_ = InterpolationType::linear;
        if (m.Get("interp", param::type::string, false, p)) {
            interp_ = StrToInterpolationType(p.str_, "AugOpticalDistortion");
        }

        border_type_ = BorderType::BORDER_REFLECT_101;
        if (m.Get("border_type", param::type::string, false, p)) {
            if (p.str_ == "constant") {
                border_type_ = BorderType::BORDER_CONSTANT;
            } else if (p.str_ == "replicate") {
                border_type_ = BorderType::BORDER_REPLICATE;
            } else if (p.str_ == "reflect") {
                border_type_ = BorderType::BORDER_REFLECT;
            } else if (p.str_ == "wrap") {
                border_type_ = BorderType::BORDER_WRAP;
            } else if (p.str_ == "reflect_101") {
                border_type_ = BorderType::BORDER_REFLECT_101;
            } else if (p.str_ == "transparent") {
                border_type_ = BorderType::BORDER_TRANSPARENT;
            } else {
                throw std::runtime_error("AugGridDistortion: invalid border type"); // TODO: standardize
            }
        }

        border_value_ = 0;
        m.Get("border_value", param::type::number, false, p);
        border_value_ = static_cast<int>(p.vals_[0]);
    }
};

/** @brief Augmentation wrapper for ecvl::Salt.

@anchor AugSalt
*/
class AugSalt : public Augmentation
{
    double per_channel_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        const auto seed = params_["seed"].value_;
        const bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        Salt(img, img, p, per_channel, static_cast<unsigned>(seed));
    }
public:
    /** @brief AugSalt constructor

    @param[in] p Parameter which determines the range of values [min,max] to randomly select the probability of any pixel being set to white.
    @param[in] per_channel Probability of each image to use the same value for all channels of a pixel or not.
    */
    AugSalt(const std::array<double, 2>& p, const double& per_channel) : per_channel_(per_channel)
    {
        assert(per_channel >= 0 && per_channel <= 1);
        params_["p"] = AugmentationParam(p[0], p[1]);
        params_["per_channel"] = AugmentationParam(0, 1);
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);
    }
    AugSalt(std::istream& is)
    {
        auto m = param::read(is, "AugSalt");
        param p;

        // seed is managed by AugmentationParam
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);

        m.Get("p", param::type::range, true, p);
        params_["p"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        m.Get("per_channel", param::type::number, true, p);
        params_["per_channel"] = AugmentationParam(0, 1);
        per_channel_ = p.vals_[0];
    }
};

/** @brief Augmentation wrapper for ecvl::Pepper.

@anchor AugPepper
*/
class AugPepper : public Augmentation
{
    double per_channel_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        const auto seed = params_["seed"].value_; 
        const bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        Pepper(img, img, p, per_channel, static_cast<unsigned>(seed));
    }
public:
    /** @brief AugPepper constructor

    @param[in] p Parameter which determines the range of values [min,max] to randomly select the probability of any pixel being set to black.
    @param[in] per_channel Probability of each image to use the same value for all channels of a pixel or not.
    */
    AugPepper(const std::array<double, 2>& p, const double& per_channel) : per_channel_(per_channel)
    {
        assert(per_channel >= 0 && per_channel <= 1);
        params_["p"] = AugmentationParam(p[0], p[1]);
        params_["per_channel"] = AugmentationParam(0, 1);
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);
    }
    AugPepper(std::istream& is)
    {
        auto m = param::read(is, "AugPepper");
        param p;

        // seed is managed by AugmentationParam
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);

        m.Get("p", param::type::range, true, p);
        params_["p"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        m.Get("per_channel", param::type::number, true, p);
        params_["per_channel"] = AugmentationParam(0, 1);
        per_channel_ = p.vals_[0];
    }
};

/** @brief Augmentation wrapper for ecvl::SaltAndPepper.

@anchor AugSaltAndPepper
*/
class AugSaltAndPepper : public Augmentation
{
    double per_channel_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        const auto p = params_["p"].value_;
        const auto seed = params_["seed"].value_; 
        const bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        SaltAndPepper(img, img, p, per_channel, static_cast<unsigned>(seed));
    }
public:
    /** @brief AugSaltAndPepper constructor

    @param[in] p Parameter which determines the range of values [min,max] to randomly select the probability of any pixel being set to white or black.
    @param[in] per_channel Probability of each image to use the same value for all channels of a pixel or not.
    */
    AugSaltAndPepper(const std::array<double, 2>& p, const double& per_channel) : per_channel_(per_channel)
    {
        assert(per_channel >= 0 && per_channel <= 1);
        params_["p"] = AugmentationParam(p[0], p[1]);
        params_["per_channel"] = AugmentationParam(0, 1);
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);
    }
    AugSaltAndPepper(std::istream& is)
    {
        auto m = param::read(is, "AugSaltAndPepper");
        param p;

        // seed is managed by AugmentationParam
        params_["seed"] = AugmentationParam(AugmentationParam::seed_min, AugmentationParam::seed_max);

        m.Get("p", param::type::range, true, p);
        params_["p"] = AugmentationParam(p.vals_[0], p.vals_[1]);

        m.Get("per_channel", param::type::number, true, p);
        params_["per_channel"] = AugmentationParam(0, 1);
        per_channel_ = p.vals_[0];
    }
};

/** @brief Augmentation wrapper for ecvl::Normalize.

@anchor AugNormalize
*/
class AugNormalize : public Augmentation
{
    double mean_ = 0., std_ = 1.;

    std::vector<double> ch_mean_;
    std::vector<double> ch_std_;

    bool per_channel_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        if (per_channel_) {
            Normalize(img, img, ch_mean_, ch_std_);
        } else {
            Normalize(img, img, mean_, std_);
        }
    }
public:
    /** @brief AugNormalize constructor

    @param[in] mean Mean to substract from all pixel.
    @param[in] std Standard deviation to use for normalization.
    */
    AugNormalize(const double& mean, const double& std) : mean_(mean), std_(std), per_channel_(false) {}

    /** @brief AugNormalize constructor with separate statistics for each channel

    @param[in] mean Per channel mean to substract from all pixel.
    @param[in] std Per channel standard deviation to use for normalization.
    */
    AugNormalize(const std::vector<double>& mean, const std::vector<double>& std) : ch_mean_(mean), ch_std_(std), per_channel_(true) {}

    AugNormalize(std::istream& is)
    {
        auto m = param::read(is, "AugNormalize");
        param p;

        m.GenericGet("mean", true, p);
        if (p.type_ == param::type::number) {
            mean_ = p.vals_[0];
            per_channel_ = false;
        } else if (p.type_ == param::type::vector) {
            ch_mean_ = p.vals_;
            per_channel_ = true;
        } else {
            throw std::runtime_error("AugNormalize: invalid mean type");
        }

        if (per_channel_ == false) {
            m.Get("std", param::type::number, true, p);
            std_ = p.vals_[0];
        } else {
            m.Get("std", param::type::vector, true, p);
            ch_std_ = p.vals_;
        }
    }
};

/** @brief Augmentation CenterCrop wrapper for ecvl::CenterCrop.

@anchor AugCenterCrop
*/
class AugCenterCrop : public Augmentation
{
    std::vector<int> size_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        CenterCrop(img, img, size_);
        if (!gt.IsEmpty()) {
            CenterCrop(gt, const_cast<Image&>(gt), size_);
        }
    }
public:
    /** @brief AugCenterCrop constructor

    @param[in] size std::vector<int> that specifies the new size of each dimension [w,h].
    */
    AugCenterCrop(const std::vector<int>& size) : size_{ size } {}

    AugCenterCrop(std::istream& is)
    {
        auto m = param::read(is, "AugCenterCrop");
        param p;

        m.Get("size", param::type::vector, true, p);
        for (const auto& x : p.vals_) {
            size_.emplace_back(static_cast<int>(x));
        }
    }
};

/** @brief Augmentation ToFloat32

This augmentation converts an Image (and ground truth) to DataType::float32 dividing it by divisor (or divisor_gt) parameter.

@anchor AugToFloat32
*/
class AugToFloat32 : public Augmentation
{
    double divisor_, divisor_gt_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        img.ConvertTo(DataType::float32);
        img.Div(divisor_);

        if (!gt.IsEmpty()) {
            const_cast<Image&>(gt).ConvertTo(DataType::float32);
            const_cast<Image&>(gt).Div(divisor_gt_);
        }
    }
public:
    /** @brief AugToFloat32 constructor

    @param[in] divisor Value used to divide the img Image.
    @param[in] divisor_gt Value used to divide the gt Image.
    */
    AugToFloat32(const double& divisor = 1., const double& divisor_gt = 1.) : divisor_{ divisor }, divisor_gt_{ divisor_gt } {}
    AugToFloat32(std::istream& is)
    {
        auto m = param::read(is, "AugToFloat32");
        param p;

        m.Get("divisor", param::type::number, false, p);
        divisor_ = p.vals_[0];
        m.Get("divisor_gt", param::type::number, false, p);
        divisor_gt_ = p.vals_[0];
    }
};

/** @brief Augmentation DivBy255

This augmentation divides an Image (and ground truth if provided) by 255.

@anchor AugDivBy255
*/
class AugDivBy255 : public Augmentation
{
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        img.Div(255);

        if (!gt.IsEmpty()) {
            const_cast<Image&>(gt).Div(255);
        }
    }
public:
    /** @brief AugDivBy255 constructor */
    AugDivBy255() {}
    AugDivBy255(std::istream& is) {}
};

/** @brief Augmentation wrapper for ecvl::AugScaleTo.

@anchor AugScaleTo
*/
class AugScaleTo : public Augmentation
{
    double new_min_, new_max_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ScaleTo(img, img, new_min_, new_max_);
    }
public:
    /** @brief AugScaleTo constructor

     @param[in] new_min double which indicates the new minimum value.
     @param[in] new_max double which indicates the new maximum value.
     */
    AugScaleTo(const double& new_min, const double& new_max) : new_min_{ new_min }, new_max_{ new_max } {}
    AugScaleTo(std::istream& is)
    {
        auto m = param::read(is, "AugScaleTo");
        param p;

        m.Get("new_min", param::type::number, true, p);
        new_min_ = p.vals_[0];

        m.Get("new_max", param::type::number, true, p);
        new_max_ = p.vals_[0];
    }
};
} // namespace ecvl

#endif // AUGMENTATIONS_H_