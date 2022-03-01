/*
* ECVL - European Computer Vision Library
* Version: 1.0.1
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <ecvl/augmentations.h>

namespace ecvl
{
std::default_random_engine AugmentationParam::re_(std::random_device{}());

param_list param::read(std::istream& is, std::string fn_name_)
{
    param_list m(move(fn_name_));
    if (is.peek() != '\n') {
        is >> std::ws;
    }
    while (is.peek() >= 0 && is.peek() != '\n') { // peek first, then check if it failed
        param p(is);
        m[p.name_] = p;
    }
    return m;
}

InterpolationType StrToInterpolationType(const std::string& interp, const std::string& aug_name)
{
    if (interp == "linear") {
        return InterpolationType::linear;
    }
    else if (interp == "area") {
        return InterpolationType::area;
    }
    else if (interp == "cubic") {
        return InterpolationType::cubic;
    }
    else if (interp == "lanczos4") {
        return InterpolationType::lanczos4;
    }
    else if (interp == "nearest") {
        return InterpolationType::nearest;
    }
    else {
        throw std::runtime_error(aug_name + ": invalid interpolation type");
    }
}

BorderType StrToBorderType(const std::string& border_type, const std::string& aug_name)
{
    if (border_type == "constant") {
        return BorderType::BORDER_CONSTANT;
    }
    else if (border_type == "reflect") {
        return BorderType::BORDER_REFLECT;
    }
    else if (border_type == "reflect101") {
        return BorderType::BORDER_REFLECT_101;
    }
    else if (border_type == "replicate") {
        return BorderType::BORDER_REPLICATE;
    }
    else if (border_type == "transparent") {
        return BorderType::BORDER_TRANSPARENT;
    }
    else if (border_type == "wrap") {
        return BorderType::BORDER_WRAP;
    }
    else {
        throw std::runtime_error(aug_name + ": invalid border type");
    }
}

// This factory must be manually populated! Don't forget to do it, otherwise no creation from streams

#define AUG(x) if (name == #x) return std::make_shared<x>(is)
std::shared_ptr<Augmentation> AugmentationFactory::create(const std::string& name, std::istream& is)
{
    AUG(SequentialAugmentationContainer);
    AUG(OneOfAugmentationContainer);
    AUG(AugRotate);
    AUG(AugResizeDim);
    AUG(AugResizeScale);
    AUG(AugFlip);
    AUG(AugMirror);
    AUG(AugGaussianBlur);
    AUG(AugAdditiveLaplaceNoise);
    AUG(AugAdditivePoissonNoise);
    AUG(AugGammaContrast);
    AUG(AugCoarseDropout);
    AUG(AugTranspose);
    AUG(AugBrightness);
    AUG(AugGridDistortion);
    AUG(AugElasticTransform);
    AUG(AugOpticalDistortion);
    AUG(AugSalt);
    AUG(AugPepper);
    AUG(AugSaltAndPepper);
    AUG(AugNormalize);
    AUG(AugCenterCrop);
    AUG(AugToFloat32);
    AUG(AugDivBy255);
    AUG(AugScaleTo);
    AUG(AugScaleFromTo);
    AUG(AugRandomCrop);

    return nullptr; // Maybe throw?
}
} // namespace ecvl