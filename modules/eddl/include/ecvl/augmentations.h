/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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

#include "ecvl/core/imgproc.h"
#include <array>
#include <map>
#include <memory>
#include <random>

namespace ecvl {
/** @brief Augmentations parameters.

This class represent the augmentations parameters which must be randomly generated in a specific range.

@anchor AugmentationParam
*/
class AugmentationParam {
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
class Augmentation {
public:
    std::map<std::string, AugmentationParam> params_;

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

// vector<move_only> cannot be constructed from initializer list :-/
// Use this work around for vector<unique_ptr<Augmentation>>
template <typename Base, typename ... Ts>
std::vector<std::unique_ptr<Base>> make_vector_of_unique(Ts&&... t)
{
    std::unique_ptr<Base> init[] = { std::make_unique<Ts>(std::forward<Ts>(t))... };
    return std::vector<std::unique_ptr<Base>> {
        std::make_move_iterator(std::begin(init)), std::make_move_iterator(std::end(init))};
}

/** @brief SequentialAugmentationContainer.

This class represent a container for multiple augmentations which will be sequentially applied to the Dataset images.

@anchor SequentialAugmentationContainer
*/
class SequentialAugmentationContainer : public Augmentation {
public:
    std::vector<std::unique_ptr<Augmentation>> augs_;   /**< @brief vector containing the Augmentation to be applied */

    SequentialAugmentationContainer() : augs_{} {}

    template<typename ...Ts>
    SequentialAugmentationContainer(Ts&&... t) : augs_(make_vector_of_unique<Augmentation>(std::forward<Ts>(t)...)) {}

    template<typename T, typename... Args>
    void Add(Args&&... args)
    {
        augs_.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

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
};

///////////////////////////////////////////////////////////////////////////////////
// Augmentations
///////////////////////////////////////////////////////////////////////////////////

/** @brief Augmentation wrapper for ecvl::Rotate2D.

@anchor AugRotate
*/
class AugRotate : public Augmentation {
    std::vector<double> center_;
    double scale_;
    InterpolationType interp_;
public:
    /** @brief AugRotate constructor

    @param[in] angle Parameter which determines the range of degrees [min,max] to randomly select from.
    @param[in] center A std::vector<double> representing the coordinates of the rotation center.
                      If empty, the center of the image is used.
    @param[in] scale Optional scaling factor.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    */
    AugRotate(const std::array<double, 2>& angle,
        const std::vector<double>& center = {},
        const double& scale = 1.,
        const InterpolationType& interp = InterpolationType::linear) : center_(center), scale_(scale), interp_(interp)
    {
        params_["angle"] = AugmentationParam(angle[0], angle[1]);
    }
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        Rotate2D(img, img, params_["angle"].value_, center_, scale_, interp_);
        if (!gt.IsEmpty()) {
            Rotate2D(gt, const_cast<Image&>(gt), params_["angle"].value_, center_, scale_, interp_);
        }
    }
};

/** @brief Augmentation wrapper for ecvl::ResizeDim.

@anchor AugResizeDim
*/
class AugResizeDim : public Augmentation {
    std::vector<int> dims_;
    InterpolationType interp_;
public:
    /** @brief AugResizeDim constructor

    @param[in] dims std::vector<int> that specifies the new size of each dimension.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    */
    AugResizeDim(const std::vector<int>& dims, const InterpolationType& interp = InterpolationType::linear) : dims_{ dims }, interp_(interp) {}

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ResizeDim(img, img, dims_, interp_);
        if (!gt.IsEmpty()) {
            ResizeDim(gt, const_cast<Image&>(gt), dims_, interp_);
        }
    }
};

/** @brief Augmentation wrapper for ecvl::ResizeScale.

@anchor AugResizeScale
*/
class AugResizeScale : public Augmentation {
    std::vector<double> scale_;
    InterpolationType interp_;
public:
    /** @brief AugResizeScale constructor

    @param[in] scale std::vector<double> that specifies the scale to apply to each dimension.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    */
    AugResizeScale(const std::vector<double>& scale, const InterpolationType& interp = InterpolationType::linear) : scale_{ scale }, interp_(interp) {}

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ResizeScale(img, img, scale_, interp_);
        if (!gt.IsEmpty()) {
            ResizeScale(gt, const_cast<Image&>(gt), scale_, interp_);
        }
    }
};

/** @brief Augmentation wrapper for ecvl::Flip2D.

@anchor AugFlip
*/
class AugFlip : public Augmentation {
    double p_;
public:
    /** @brief AugFlip constructor

    @param[in] p Probability of each image to get flipped.
    */
    AugFlip(const double& p) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        if (params_["p"].value_ <= p_) {
            Flip2D(img, img);
            if (!gt.IsEmpty()) {
                Flip2D(gt, const_cast<Image&>(gt));
            }
        }
    }
};

/** @brief Augmentation wrapper for ecvl::Mirror2D.

@anchor AugMirror
*/
class AugMirror : public Augmentation {
    double p_;
public:
    /** @brief AugFlip constructor

    @param[in] p Probability of each image to get mirrored.
    */
    AugMirror(const double& p) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        if (params_["p"].value_ <= p_) {
            Mirror2D(img, img);
            if (!gt.IsEmpty()) {
                Mirror2D(gt, const_cast<Image&>(gt));
            }
        }
    }
};

/** @brief Augmentation wrapper for ecvl::GaussianBlur.

@anchor AugGaussianBlur
*/
class AugGaussianBlur : public Augmentation {
public:
    /** @brief AugGaussianBlur constructor

    @param[in] sigma Parameter which determines the range of sigma [min,max] to randomly select from.
    */
    AugGaussianBlur(const std::array<double, 2>& sigma)
    {
        params_["sigma"] = AugmentationParam(sigma[0], sigma[1]);
    }

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        GaussianBlur(img, img, params_["sigma"].value_);
    }
};

/** @brief Augmentation wrapper for ecvl::AdditiveLaplaceNoise.

@anchor AugAdditiveLaplaceNoise
*/
class AugAdditiveLaplaceNoise : public Augmentation {
public:
    /** @brief AugAdditiveLaplaceNoise constructor

    @param[in] std_dev Parameter which determines the range of values [min,max] to randomly select the standard deviation of the noise generating distribution.
                       Suggested values are around 255 * 0.05 for uint8 Images.
    */
    AugAdditiveLaplaceNoise(const std::array<double, 2>& std_dev)
    {
        params_["std_dev"] = AugmentationParam(std_dev[0], std_dev[1]);
    }

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        AdditiveLaplaceNoise(img, img, params_["std_dev"].value_);
    }
};

/** @brief Augmentation wrapper for ecvl::AdditivePoissonNoise.

@anchor AugAdditivePoissonNoise
*/
class AugAdditivePoissonNoise : public Augmentation {
public:
    /** @brief AugAdditivePoissonNoise constructor

    @param[in] lambda Parameter which determines the range of values [min,max] to randomly select the lambda of the noise generating distribution.
                      Suggested values are around 0.0 to 10.0.
    */
    AugAdditivePoissonNoise(const std::array<double, 2>& lambda)
    {
        params_["lambda"] = AugmentationParam(lambda[0], lambda[1]);
    }

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        AdditivePoissonNoise(img, img, params_["lambda"].value_);
    }
};

/** @brief Augmentation wrapper for ecvl::GammaContrast.

@anchor AugGammaContrast
*/
class AugGammaContrast : public Augmentation {
public:
    /** @brief AugGammaContrast constructor

    @param[in] gamma Parameter which determines the range of values [min,max] to randomly select the exponent for the contrast adjustment.
                     Suggested values are around 0.5 to 2.0.
    */
    AugGammaContrast(const std::array<double, 2>& gamma)
    {
        params_["gamma"] = AugmentationParam(gamma[0], gamma[1]);
    }

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        GammaContrast(img, img, params_["gamma"].value_);
    }
};

/** @brief Augmentation wrapper for ecvl::CoarseDropout.

@anchor AugCoarseDropout
*/
class AugCoarseDropout : public Augmentation {
    double per_channel_;
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

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        CoarseDropout(img, img, params_["p"].value_, params_["drop_size"].value_, per_channel);
    }
};
} // namespace ecvl

#endif // AUGMENTATIONS_H_