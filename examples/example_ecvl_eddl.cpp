/*
* ECVL - European Computer Vision Library
* Version: 1.0.3
* copyright (c) 2021, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <iostream>
#include <sstream>

#include "ecvl/core.h"
#include "ecvl/support_eddl.h"
#include "ecvl/augmentations.h"

using namespace ecvl;
using namespace eddl;
using namespace std;

int main()
{
    // Open an Image
    Image img, tmp;
    if (!ImRead("../examples/data/test.jpg", img)) {
        return EXIT_FAILURE;
    }
    tmp = img;

    // Create an augmentation sequence to be applied to the image
    auto augs = make_shared<SequentialAugmentationContainer>(
        AugCenterCrop(), // Make image squared
        AugRotate({ -5, 5 }),
        AugMirror(.5),
        AugFlip(.5),
        AugGammaContrast({ 3, 5 }),
        AugAdditiveLaplaceNoise({ 0, 0.2 * 255 }),
        AugCoarseDropout({ 0, 0.55 }, { 0.02, 0.1 }, 0.5),
        AugAdditivePoissonNoise({ 0, 40 }),
        AugResizeDim({ 500, 500 }),
        AugCenterCrop({ 224, 224 }),
        AugToFloat32(255),
        AugNormalize({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 })
        );

    // Replace the random seed with a fixed one
    AugmentationParam::SetSeed(0);

    // Apply the augmentations
    augs->Apply(img);

    // Convert an Image into tensor
    cout << "Executing ImageToTensor" << endl;
    Tensor* t;
    ImageToTensor(img, t);

    // Apply eddl functions
    t->div_(128);
    t->mult_(128);

    // Convert a tensor into an Image
    cout << "Executing TensorToImage" << endl;
    TensorToImage(t, img);

    // Convert a tensor into a View (they point to the same data)
    View<DataType::float32> view;
    cout << "Executing TensorToView" << endl;
    TensorToView(t, view);

    // Create an augmentation sequence from stream
    stringstream ss(
        "SequentialAugmentationContainer\n"
        "    AugRotate angle=[-5,5] center=(0,0) interp=\"linear\"\n"
        "    AugAdditiveLaplaceNoise std_dev=[0,0.51]\n"
        "    AugCoarseDropout p=[0,0.55] drop_size=[0.02,0.1] per_channel=0\n"
        "    AugAdditivePoissonNoise lambda=[0,40]\n"
        "    AugResizeDim dims=(224,224) interp=\"linear\"\n"
        "    AugToFloat32 divisor=255\n"
        "    AugNormalize mean=(0.485, 0.456, 0.406) std=(0.229, 0.224, 0.225)\n"
        "end\n"
    );
    auto newdeal_augs = AugmentationFactory::create(ss);
    newdeal_augs->Apply(tmp);

    /*--------------------------------------------------------------------------------------------*/

    // Create the augmentations to be applied to the dataset images during training and test.
    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugRotate({ -5, 5 }),
        AugAdditiveLaplaceNoise({ 0, 0.2 * 255 }),
        AugCoarseDropout({ 0, 0.55 }, { 0.02,0.1 }, 0),
        AugAdditivePoissonNoise({ 0, 40 }),
        AugResizeDim({ 30, 30 }),
        AugToFloat32(255),
        AugNormalize({ 0.449 }, { 0.226 }) // mean of imagenet stats
        );

    auto test_augs = make_shared<SequentialAugmentationContainer>(
        AugResizeDim({ 30, 30 }),
        AugToFloat32(255),
        AugNormalize({ 0.449 }, { 0.226 }) // mean of imagenet stats
        );

    // OLD version: now the number of augmentations must match the number of splits in the yml file
    // DatasetAugmentations dataset_augmentations{ {training_augs, nullptr, test_augs } };
    DatasetAugmentations dataset_augmentations{ {training_augs, test_augs } };

    int batch_size = 64;
    cout << "Creating a DLDataset" << endl;
    DLDataset d("../examples/data/mnist/mnist.yml", batch_size, dataset_augmentations, ColorType::GRAY);

    // Allocate memory for x and y tensors
    cout << "Create x and y" << endl;
    Tensor* x = new Tensor({ batch_size, d.n_channels_, d.resize_dims_[0], d.resize_dims_[1] });
    Tensor* y = new Tensor({ batch_size, static_cast<int>(d.classes_.size()) });

    // Load a batch of d.batch_size_ images into x and corresponding labels in y
    // Images are resized to the dimensions specified in the augmentations chain
    cout << "Executing LoadBatch on training set" << endl;
    d.LoadBatch(x, y);

    // Change colortype and channels for Image Watch visualization
    TensorToImage(x, img);
    img.colortype_ = ColorType::GRAY;
    img.channels_ = "xyc";

    // Extract the first plane from img
    View<DataType::float32> img_v(img, { 0,0,0 }, { -1,-1,1 });

    // Switch to Test split and load a batch of images in x and corresponding labels in y
    cout << "Executing LoadBatch on test set" << endl;
    d.SetSplit(SplitType::test);
    d.LoadBatch(x, y);

    // Save some input images
    ImWrite("mnist_batch.png", MakeGrid(x, 8, false));
    ImWrite("mnist_batch_normalized.png", MakeGrid(x, 8, true));

    delete x;
    delete y;
    delete t;

    return EXIT_SUCCESS;
}