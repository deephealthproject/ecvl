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
#include <random>

namespace ecvl {
class AugmentationParam {
public:
    double min_, max_, value_;

    AugmentationParam() = default;
    AugmentationParam(double min, double max) : min_(min), max_(max) {}
    void GenerateValue()
    {
        value_ = std::uniform_real_distribution<>(min_, max_)(re_);
    }
    static std::default_random_engine re_;
    static void SetSeed(unsigned seed)
    {
        re_.seed(seed);
    }
};

class Augmentation {
public:
    std::map<std::string, AugmentationParam> params_;
    void Apply(ecvl::Image& img)
    {
        for (auto& x : params_) {
            x.second.GenerateValue();
        }
        RealApply(img);
    }
    virtual ~Augmentation() = default;
private:
    virtual void RealApply(ecvl::Image& img) = 0;
};

// vector<move_only> cannot be construct from initializer list :-/
// Use this work around
template <typename Base, typename ... Ts>
std::vector<std::unique_ptr<Base>> make_vector_of_unique(Ts&&... t)
{
    std::unique_ptr<Base> init[] = { make_unique<Ts>(std::forward<Ts>(t))... };
    return std::vector<std::unique_ptr<Base>> {
        std::make_move_iterator(std::begin(init)),
            std::make_move_iterator(std::end(init))};
}

class SequentialAugmentationContainer : public Augmentation {
public:
    std::vector<std::unique_ptr<Augmentation>> augs_;

    SequentialAugmentationContainer() : augs_{} {}

    template<typename ...Ts>
    SequentialAugmentationContainer(Ts&&... t) : augs_(make_vector_of_unique<Augmentation>(std::forward<Ts>(t)...)) {}

    template<typename T, typename... Args>
    void Add(Args&&... args)
    {
        augs_.emplace_back(make_unique<T>(std::forward<Args>(args)...));
    }

    virtual void RealApply(ecvl::Image& img) override
    {
        for (auto& x : augs_) {
            x->Apply(img);
        }
    }
};

class AugRotate : public Augmentation {
    std::vector<double> center_;
    double scale_;
    InterpolationType interp_;
public:
    AugRotate(std::array<double, 2> angle,
        std::vector<double> center = {},
        double scale = 1.,
        InterpolationType interp = InterpolationType::linear) : center_(std::move(center)), scale_(scale), interp_(interp)
    {
        params_["angle"] = AugmentationParam(angle[0], angle[1]);
    }
    virtual void RealApply(ecvl::Image& img) override
    {
        Rotate2D(img, img, params_["angle"].value_, center_, scale_, interp_);
    }
};

class AugRotateFullImage : public Augmentation {
    double scale_;
    InterpolationType interp_;
public:
    AugRotateFullImage(std::array<double, 2> angle, double scale = 1., InterpolationType interp = InterpolationType::linear) : scale_(scale), interp_(interp)
    {
        params_["angle"] = AugmentationParam(angle[0], angle[1]);
    }
    virtual void RealApply(ecvl::Image& img) override
    {
        RotateFullImage2D(img, img, params_["angle"].value_, scale_, interp_);
    }
};

class AugResizeDim : public Augmentation {
    std::vector<int> dims_;
    InterpolationType interp_;
public:

    AugResizeDim(std::vector<int> dims, InterpolationType interp = InterpolationType::linear) : dims_{ std::move(dims) }, interp_(interp) {}

    virtual void RealApply(ecvl::Image& img) override
    {
        ResizeDim(img, img, dims_, interp_);
    }
};

class AugResizeScale : public Augmentation {
    std::vector<double> scale_;
    InterpolationType interp_;
public:

    AugResizeScale(std::vector<double> scale, InterpolationType interp = InterpolationType::linear) : scale_{ std::move(scale) }, interp_(interp) {}

    virtual void RealApply(ecvl::Image& img) override
    {
        ResizeScale(img, img, scale_, interp_);
    }
};

class AugFlip : public Augmentation {
    double p_;
public:
    AugFlip(double p) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }
    virtual void RealApply(ecvl::Image& img) override
    {
        if (params_["p"].value_ <= p_) {
            Flip2D(img, img);
        }
    }
};

class AugMirror : public Augmentation {
    double p_;
public:
    AugMirror(double p) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }
    virtual void RealApply(ecvl::Image& img) override
    {
        if (params_["p"].value_ <= p_) {
            Mirror2D(img, img);
        }
    }
};

class AugGaussianBlur : public Augmentation {
public:

    AugGaussianBlur(std::array<double, 2> sigma)
    {
        params_["sigma"] = AugmentationParam(sigma[0], sigma[1]);
    }

    virtual void RealApply(ecvl::Image& img) override
    {
        GaussianBlur(img, img, params_["sigma"].value_);
    }
};

class AugAdditiveLaplaceNoise : public Augmentation {
public:

    AugAdditiveLaplaceNoise(std::array<double, 2> std_dev)
    {
        params_["std_dev"] = AugmentationParam(std_dev[0], std_dev[1]);
    }

    virtual void RealApply(ecvl::Image& img) override
    {
        AdditiveLaplaceNoise(img, img, params_["std_dev"].value_);
    }
};

class AugAdditivePoissonNoise : public Augmentation {
public:

    AugAdditivePoissonNoise(std::array<double, 2> lambda)
    {
        params_["lambda"] = AugmentationParam(lambda[0], lambda[1]);
    }

    virtual void RealApply(ecvl::Image& img) override
    {
        AdditivePoissonNoise(img, img, params_["lambda"].value_);
    }
};

class AugGammaContrast : public Augmentation {
public:

    AugGammaContrast(std::array<double, 2> gamma)
    {
        params_["gamma"] = AugmentationParam(gamma[0], gamma[1]);
    }

    virtual void RealApply(ecvl::Image& img) override
    {
        GammaContrast(img, img, params_["gamma"].value_);
    }
};

class AugCoarseDropout : public Augmentation {
    double per_channel_;
public:

    AugCoarseDropout(std::array<double, 2> p, std::array<double, 2> drop_size, double per_channel) : per_channel_(per_channel)
    {
        assert(per_channel >= 0 && per_channel <= 1);
        params_["p"] = AugmentationParam(p[0], p[1]);
        params_["drop_size"] = AugmentationParam(drop_size[0], drop_size[1]);
        params_["per_channel"] = AugmentationParam(0, 1);
    }

    virtual void RealApply(ecvl::Image& img) override
    {
        bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        CoarseDropout(img, img, params_["p"].value_, params_["drop_size"].value_, per_channel);
    }
};
} // namespace ecvl

#endif // AUGMENTATIONS_H_