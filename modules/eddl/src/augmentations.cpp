/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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

param_list param::read(std::istream& is, std::string fn_name_) {
    param_list m(move(fn_name_));
    is >> std::ws;
    while (is.peek() >= 0 && is.peek() != '\n') { // peek first, then check if it failed
        param p(is);
        m[p.name_] = p;
    }
    return m;
}

// This factory must be manually populated! Don't forget to do it, otherwise no creation from streams

#define AUG(x) if (name == #x) return std::make_shared<x>(is)
std::shared_ptr<Augmentation> AugmentationFactory::create(const std::string& name, std::istream& is) 
{
    AUG(SequentialAugmentationContainer);
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

    return nullptr; // Maybe throw?
}

} // namespace ecvl