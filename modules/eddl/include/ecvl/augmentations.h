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
#include <iostream>
#include <algorithm>
#include <iterator>
#include <unordered_map>

namespace ecvl {

#define ECVL_ERROR_AUGMENTATION_NAME throw std::runtime_error(ECVL_ERROR_MSG "Cannot load augmentation name");
#define ECVL_ERROR_AUGMENTATION_FORMAT throw std::runtime_error(ECVL_ERROR_MSG "Format error while loading augmentation parameters");

class param {
	static std::istream& read_until(std::istream& is, std::string& s, const std::string& list) {
		s.clear();
		while (is.peek() && is && list.find(is.peek()) == list.npos) {
			s += is.get();
		}
		return is;
	}

	void read_vals(std::istream& is, char closing_char) {
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

    static const char* to_string(type t) {
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
	param(std::istream& is) {
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
		}
		else if (next_char == '(') { // vector
			is.ignore();
			type_ = type::vector;
			read_vals(is, ')');
		}
		else if (next_char == '"') { // string
			is.ignore();
			type_ = type::string;
			std::getline(is, str_, '"');
		}
		else {
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

class param_list {
    std::unordered_map<std::string, param> m_;
    const std::string fn_name_;
public:
    param_list(std::string fn_name) : fn_name_(move(fn_name)) {}

    auto& operator[](const std::string& s) {
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
};

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

struct AugmentationFactory {
    static std::shared_ptr<Augmentation> create(std::istream& is) {
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

This class represent a container for multiple augmentations which will be sequentially applied to the Dataset images.

@anchor SequentialAugmentationContainer
*/
class SequentialAugmentationContainer : public Augmentation {
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

	SequentialAugmentationContainer(std::istream& is) {
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

	virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
	{
		Rotate2D(img, img, params_["angle"].value_, center_, scale_, interp_);
		if (!gt.IsEmpty()) {
			Rotate2D(gt, const_cast<Image&>(gt), params_["angle"].value_, center_, scale_, interp_);
		}
	}
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

	AugRotate(std::istream& is) {
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
        if (m.Get("interp", param::type::string, false, p)) {
			if (p.str_ == "linear") {
				interp_ = InterpolationType::linear;
			}
			else if (p.str_ == "area") {
				interp_ = InterpolationType::area;
			}
			else if (p.str_ == "cubic") {
				interp_ = InterpolationType::cubic;
			}
			else if (p.str_ == "lanczos4") {
				interp_ = InterpolationType::lanczos4;
			}
			else if (p.str_ == "nearest") {
				interp_ = InterpolationType::nearest;
			}
			else {
				throw std::runtime_error("AugRotate: invalid interpolation type"); // TODO: standardize
			}
		}
	}
};

/** @brief Augmentation wrapper for ecvl::ResizeDim.

@anchor AugResizeDim
*/
class AugResizeDim : public Augmentation {
    std::vector<int> dims_;
    InterpolationType interp_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ResizeDim(img, img, dims_, interp_);
        if (!gt.IsEmpty()) {
            ResizeDim(gt, const_cast<Image&>(gt), dims_, interp_);
        }
    }
public:
    /** @brief AugResizeDim constructor

    @param[in] dims std::vector<int> that specifies the new size of each dimension.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    */
    AugResizeDim(const std::vector<int>& dims, const InterpolationType& interp = InterpolationType::linear) : dims_{ dims }, interp_(interp) {}

	AugResizeDim(std::istream& is) {
        auto m = param::read(is, "AugResizeDim");
        param p;

        m.Get("dims", param::type::vector, true, p);
        for (const auto& x : p.vals_) {
            dims_.emplace_back(static_cast<int>(x));
        }

        interp_ = InterpolationType::linear;
        if (m.Get("interp", param::type::string, false, p)) {
            if (p.str_ == "linear") {
                interp_ = InterpolationType::linear;
            }
            else if (p.str_ == "area") {
                interp_ = InterpolationType::area;
            }
            else if (p.str_ == "cubic") {
                interp_ = InterpolationType::cubic;
            }
            else if (p.str_ == "lanczos4") {
                interp_ = InterpolationType::lanczos4;
            }
            else if (p.str_ == "nearest") {
                interp_ = InterpolationType::nearest;
            }
            else {
                throw std::runtime_error("AugRotate: invalid interpolation type"); // TODO: standardize
            }
        }
	}
};

/** @brief Augmentation wrapper for ecvl::ResizeScale.

@anchor AugResizeScale
*/
class AugResizeScale : public Augmentation {
    std::vector<double> scale_;
    InterpolationType interp_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        ResizeScale(img, img, scale_, interp_);
        if (!gt.IsEmpty()) {
            ResizeScale(gt, const_cast<Image&>(gt), scale_, interp_);
        }
    }
public:
    /** @brief AugResizeScale constructor

    @param[in] scale std::vector<double> that specifies the scale to apply to each dimension.
    @param[in] interp InterpolationType to be used. Default is InterpolationType::linear.
    */
    AugResizeScale(const std::vector<double>& scale, const InterpolationType& interp = InterpolationType::linear) : scale_{ scale }, interp_(interp) {}

	AugResizeScale(std::istream& is) {
		auto m = param::read(is, "AugResizeScale");
        param p;

        m.Get("scale", param::type::vector, true, p);
		scale_ = p.vals_;

        interp_ = InterpolationType::linear;
        if (m.Get("interp", param::type::string, false, p)) {
            if (p.str_ == "linear") {
                interp_ = InterpolationType::linear;
            }
            else if (p.str_ == "area") {
                interp_ = InterpolationType::area;
            }
            else if (p.str_ == "cubic") {
                interp_ = InterpolationType::cubic;
            }
            else if (p.str_ == "lanczos4") {
                interp_ = InterpolationType::lanczos4;
            }
            else if (p.str_ == "nearest") {
                interp_ = InterpolationType::nearest;
            }
            else {
                throw std::runtime_error("AugRotate: invalid interpolation type"); // TODO: standardize
            }
        }
    }
};

/** @brief Augmentation wrapper for ecvl::Flip2D.

@anchor AugFlip
*/
class AugFlip : public Augmentation {
    double p_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        if (params_["p"].value_ <= p_) {
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

	AugFlip(std::istream& is) : AugFlip() {
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
class AugMirror : public Augmentation {
    double p_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        if (params_["p"].value_ <= p_) {
            Mirror2D(img, img);
            if (!gt.IsEmpty()) {
                Mirror2D(gt, const_cast<Image&>(gt));
            }
        }
    }
public:
    /** @brief AugFlip constructor

    @param[in] p Probability of each image to get mirrored.
    */
    AugMirror(double p = 0.5) : p_{ p }
    {
        params_["p"] = AugmentationParam(0, 1);
    }

	AugMirror(std::istream& is) : AugMirror() {
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
class AugGaussianBlur : public Augmentation {
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        GaussianBlur(img, img, params_["sigma"].value_);
    }
public:
    /** @brief AugGaussianBlur constructor

    @param[in] sigma Parameter which determines the range of sigma [min,max] to randomly select from.
    */
    AugGaussianBlur(const std::array<double, 2>& sigma)
    {
        params_["sigma"] = AugmentationParam(sigma[0], sigma[1]);
    }

	AugGaussianBlur(std::istream& is) {
        auto m = param::read(is, "AugGaussianBlur");
        param p;

        m.Get("sigma", param::type::range, true, p);
		params_["sigma"] = AugmentationParam(p.vals_[0], p.vals_[1]);
	}
};

/** @brief Augmentation wrapper for ecvl::AdditiveLaplaceNoise.

@anchor AugAdditiveLaplaceNoise
*/
class AugAdditiveLaplaceNoise : public Augmentation {
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        AdditiveLaplaceNoise(img, img, params_["std_dev"].value_);
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

	AugAdditiveLaplaceNoise(std::istream& is) {
        auto m = param::read(is, "AugAdditiveLaplaceNoise");
        param p;

        m.Get("std_dev", param::type::range, true, p);
        params_["std_dev"] = AugmentationParam(p.vals_[0], p.vals_[1]);
	}
};

/** @brief Augmentation wrapper for ecvl::AdditivePoissonNoise.

@anchor AugAdditivePoissonNoise
*/
class AugAdditivePoissonNoise : public Augmentation {
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        AdditivePoissonNoise(img, img, params_["lambda"].value_);
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

	AugAdditivePoissonNoise(std::istream& is) {
        auto m = param::read(is, "AugAdditivePoissonNoise");
        param p;

        m.Get("lambda", param::type::range, true, p);
        params_["lambda"] = AugmentationParam(p.vals_[0], p.vals_[1]);
    }
};

/** @brief Augmentation wrapper for ecvl::GammaContrast.

@anchor AugGammaContrast
*/
class AugGammaContrast : public Augmentation {
    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        GammaContrast(img, img, params_["gamma"].value_);
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

	AugGammaContrast(std::istream& is) {
        auto m = param::read(is, "AugGammaContrast");
        param p;

        m.Get("gamma", param::type::range, true, p);
        params_["gamma"] = AugmentationParam(p.vals_[0], p.vals_[1]);
	}
};

/** @brief Augmentation wrapper for ecvl::CoarseDropout.

@anchor AugCoarseDropout
*/
class AugCoarseDropout : public Augmentation {
    double per_channel_;

    virtual void RealApply(ecvl::Image& img, const ecvl::Image& gt = Image()) override
    {
        bool per_channel = params_["per_channel"].value_ <= per_channel_ ? true : false;
        CoarseDropout(img, img, params_["p"].value_, params_["drop_size"].value_, per_channel);
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
	AugCoarseDropout(std::istream& is) {
        auto m = param::read(is, "AugGammaContrast");
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

} // namespace ecvl

#endif // AUGMENTATIONS_H_