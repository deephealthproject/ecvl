 /*
* ECVL - European Computer Vision Library
* Version: 0.3.1
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
    ss = std::stringstream("p=0.4");
    EXPECT_NO_THROW(p = make_unique<AugTranspose>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("beta=[30,60]");
    EXPECT_NO_THROW(p = make_unique<AugBrightness>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("num_steps=[5,10] distort_limit=[-0.2,0.2] interp=\"linear\" border_type=\"reflect_101\" border_value=0");
    EXPECT_NO_THROW(p = make_unique<AugGridDistortion>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("alpha=[34,60] sigma=[4,6] interp=\"linear\" border_type=\"reflect_101\" border_value=0");
    EXPECT_NO_THROW(p = make_unique<AugElasticTransform>(ss));
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
    ss = std::stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugTranspose>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("beta=[30,60]");
    EXPECT_NO_THROW(p = make_unique<AugBrightness>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("num_steps=[5,10] distort_limit=[-0.2,0.2]");
    EXPECT_NO_THROW(p = make_unique<AugGridDistortion>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = std::stringstream("alpha=[34,60] sigma=[4,6]");
    EXPECT_NO_THROW(p = make_unique<AugElasticTransform>(ss));
    EXPECT_NO_THROW(p->Apply(img));
}

TEST(Augmentations, ConstructFromStreamWithWrongParms)
{
    Image img({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    std::unique_ptr<Augmentation> p;
    std::stringstream ss("angle=(-5,5)");
    EXPECT_THROW(p = make_unique<AugRotate>(ss), std::runtime_error);
    ss = std::stringstream("dims=100");
    EXPECT_THROW(p = make_unique<AugResizeDim>(ss), std::runtime_error);
    ss = std::stringstream("");
    EXPECT_THROW(p = make_unique<AugResizeScale>(ss), std::runtime_error);
    ss = std::stringstream("p=\"test\"");
    EXPECT_THROW(p = make_unique<AugFlip>(ss), std::runtime_error);
    ss = std::stringstream("");
    EXPECT_THROW(p = make_unique<AugBrightness>(ss), std::runtime_error);
    ss = std::stringstream("num_steps=[5,10] distort_limit=(-0.2,0.2)");
    EXPECT_THROW(p = make_unique<AugGridDistortion>(ss), std::runtime_error);
    ss = std::stringstream("alpha=34");
    EXPECT_THROW(p = make_unique<AugElasticTransform>(ss), std::runtime_error);
}