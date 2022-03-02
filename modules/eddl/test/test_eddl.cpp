/*
* ECVL - European Computer Vision Library
* Version: 1.0.2
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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
using namespace std;
using std::stringstream;
using std::unique_ptr;
using std::runtime_error;

TEST(Augmentations, ConstructFromStreamAllParamsOk)
{
    Image img({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    unique_ptr<Augmentation> p;
    stringstream ss("angle=[-5,5] center=(0,0) scale=0.5 interp=\"linear\"");
    EXPECT_NO_THROW(p = make_unique<AugRotate>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("dims=(100,100) interp=\"linear\"");
    EXPECT_NO_THROW(p = make_unique<AugResizeDim>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("scale=(1.,2.) interp=\"linear\"");
    EXPECT_NO_THROW(p = make_unique<AugResizeScale>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=0.3");
    EXPECT_NO_THROW(p = make_unique<AugFlip>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=0.3");
    EXPECT_NO_THROW(p = make_unique<AugMirror>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("sigma=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugGaussianBlur>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("std_dev=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugAdditiveLaplaceNoise>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("lambda=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugAdditivePoissonNoise>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("gamma=[1.,2.]");
    EXPECT_NO_THROW(p = make_unique<AugGammaContrast>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] drop_size=[0.02,0.1] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugCoarseDropout>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=0.4");
    EXPECT_NO_THROW(p = make_unique<AugTranspose>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("beta=[30,60]");
    EXPECT_NO_THROW(p = make_unique<AugBrightness>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("num_steps=[5,10] distort_limit=[-0.2,0.2] interp=\"linear\" border_type=\"reflect_101\" border_value=0");
    EXPECT_NO_THROW(p = make_unique<AugGridDistortion>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("alpha=[34,60] sigma=[4,6] interp=\"linear\" border_type=\"reflect_101\" border_value=0");
    EXPECT_NO_THROW(p = make_unique<AugElasticTransform>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("distort_limit=[5,10] shift_limit=[4,6] interp=\"linear\" border_type=\"reflect_101\" border_value=0");
    EXPECT_NO_THROW(p = make_unique<AugOpticalDistortion>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugSalt>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugPepper>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugSaltAndPepper>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("mean=100 std=1");
    EXPECT_NO_THROW(p = make_unique<AugNormalize>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugCenterCrop>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("size=(100,100)");
    EXPECT_NO_THROW(p = make_unique<AugCenterCrop>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("divisor=255 divisor_gt=255");
    EXPECT_NO_THROW(p = make_unique<AugToFloat32>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugDivBy255>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("new_min=0 new_max=1");
    EXPECT_NO_THROW(p = make_unique<AugScaleTo>(ss));
    EXPECT_NO_THROW(p->Apply(img));
}

TEST(Augmentations, ConstructFromStreamWithoutOptionalParms)
{
    Image img({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    unique_ptr<Augmentation> p;
    stringstream ss("angle=[-5,5]");
    EXPECT_NO_THROW(p = make_unique<AugRotate>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("dims=(100,100)");
    EXPECT_NO_THROW(p = make_unique<AugResizeDim>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("scale=(1.,2.)");
    EXPECT_NO_THROW(p = make_unique<AugResizeScale>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugFlip>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugMirror>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugTranspose>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("beta=[30,60]");
    EXPECT_NO_THROW(p = make_unique<AugBrightness>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("num_steps=[5,10] distort_limit=[-0.2,0.2]");
    EXPECT_NO_THROW(p = make_unique<AugGridDistortion>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("alpha=[34,60] sigma=[4,6]");
    EXPECT_NO_THROW(p = make_unique<AugElasticTransform>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("distort_limit=[5,10] shift_limit=[4,6]");
    EXPECT_NO_THROW(p = make_unique<AugOpticalDistortion>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugSalt>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugPepper>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("p=[0,0.55] per_channel=0");
    EXPECT_NO_THROW(p = make_unique<AugSaltAndPepper>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("mean=100 std=1");
    EXPECT_NO_THROW(p = make_unique<AugNormalize>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("size=(100,100)");
    EXPECT_NO_THROW(p = make_unique<AugCenterCrop>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("divisor=255");
    EXPECT_NO_THROW(p = make_unique<AugToFloat32>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("");
    EXPECT_NO_THROW(p = make_unique<AugDivBy255>(ss));
    EXPECT_NO_THROW(p->Apply(img));
    ss = stringstream("new_min=0 new_max=1");
    EXPECT_NO_THROW(p = make_unique<AugScaleTo>(ss));
    EXPECT_NO_THROW(p->Apply(img));
}

TEST(Augmentations, ConstructFromStreamWithWrongParms)
{
    Image img({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    unique_ptr<Augmentation> p;
    stringstream ss("angle=(-5,5)");
    EXPECT_THROW(p = make_unique<AugRotate>(ss), runtime_error);
    ss = stringstream("dims=100");
    EXPECT_THROW(p = make_unique<AugResizeDim>(ss), runtime_error);
    ss = stringstream("");
    EXPECT_THROW(p = make_unique<AugResizeScale>(ss), runtime_error);
    ss = stringstream("p=\"test\"");
    EXPECT_THROW(p = make_unique<AugFlip>(ss), runtime_error);
    ss = stringstream("");
    EXPECT_THROW(p = make_unique<AugBrightness>(ss), runtime_error);
    ss = stringstream("num_steps=[5,10] distort_limit=(-0.2,0.2)");
    EXPECT_THROW(p = make_unique<AugGridDistortion>(ss), runtime_error);
    ss = stringstream("alpha=34");
    EXPECT_THROW(p = make_unique<AugElasticTransform>(ss), runtime_error);
}