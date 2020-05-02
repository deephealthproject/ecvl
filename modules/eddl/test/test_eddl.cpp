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

#include <gtest/gtest.h>

#include <sstream>

#include "ecvl/support_eddl.h"
#include "ecvl/augmentations.h"

using namespace ecvl;

TEST(Augmentations, ConstructFromStreamAllParamsOk)
{
    Image img({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    std::unique_ptr<Augmentation> p;
    std::stringstream ss("angle=[-5,5] center=(0,0) scale=0.5 interp=\"linear\"");
    EXPECT_NO_THROW(p = make_unique<AugRotate>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("dims=(100,100) interp=\"linear\"");
    EXPECT_NO_THROW(p = make_unique<AugResizeDim>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("scale=(1.,2.) interp=\"linear\"");
    EXPECT_NO_THROW(p = make_unique<AugResizeScale>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("p=0.3");
    EXPECT_NO_THROW(p = make_unique<AugFlip>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("p=0.3");
    EXPECT_NO_THROW(p = make_unique<AugMirror>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("sigma=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugGaussianBlur>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("std_dev=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugAdditiveLaplaceNoise>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("lambda=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugAdditivePoissonNoise>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("gamma=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugGammaContrast>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("p=[0,0.55] drop_size=[0.02,0.1] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugCoarseDropout>(ss));
    EXPECT_NO_THROW(p->Apply(img));
}

TEST(Augmentations, ConstructFromStreamWithoutOptionalParms)
{
    Image img({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    std::unique_ptr<Augmentation> p;
    std::stringstream ss("angle=[-5,5]");
    EXPECT_NO_THROW(p = make_unique<AugRotate>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("dims=(100,100)");
    EXPECT_NO_THROW(p = make_unique<AugResizeDim>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("scale=(1.,2.)");
    EXPECT_NO_THROW(p = make_unique<AugResizeScale>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugFlip>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugMirror>(ss));
    EXPECT_NO_THROW(p->Apply(img));
}

