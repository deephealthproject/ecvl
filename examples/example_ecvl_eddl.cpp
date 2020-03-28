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

#include <iostream>
#include "ecvl/core.h"
#include "ecvl/support_eddl.h"
#include "ecvl/augmentations.h"

using namespace ecvl;
using namespace eddl;
using namespace std;

int main()
{
    // Open an Image
    Image img;
    if (!ImRead("../examples/data/test.jpg", img)) {
        return EXIT_FAILURE;
    }

    // Create an augmentation sequence to be applied to the image
    auto augs = make_unique<SequentialAugmentationContainer>(
        AugRotate({ -5, 5 }),
        AugMirror(.5),
        AugFlip(.5),
        AugGammaContrast({ 3, 5 }),
        AugAdditiveLaplaceNoise({ 0, 0.2 * 255 }),
        AugCoarseDropout({ 0, 0.55 }, { 0.02,0.1 }, 0.5),
        AugAdditivePoissonNoise({ 0, 40 }),
        AugResizeDim({ 500, 500 })
        );

    // Replace the random seed with a fixed one
    AugmentationParam::SetSeed(0);

    // Apply the augmentations
    augs->Apply(img);

    // Convert an Image into tensor
    cout << "Executing ImageToTensor" << endl;
    tensor t;
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

    // Create the augmentations to be applied to the dataset images during training and test.
    // nullptr is given as augmentation for validation because this split doesn't exist in the mnist dataset.
    auto training_augs = make_unique<SequentialAugmentationContainer>(
        AugRotate({ -5, 5 }),
        AugAdditiveLaplaceNoise({ 0, 0.2 * 255 }),
        AugCoarseDropout({ 0, 0.55 }, { 0.02,0.1 }, 0),
        AugAdditivePoissonNoise({ 0, 40 }),
        AugResizeDim({ 30, 30 })
        );

    auto test_augs = make_unique<SequentialAugmentationContainer>(
        AugResizeDim({ 30, 30 })
        );
	
    DatasetAugmentations dataset_augmentations{ {move(training_augs), nullptr, move(test_augs) } };

    int batch_size = 64;
    cout << "Creating a DLDataset" << endl;
    DLDataset d("../examples/data/mnist/mnist.yml", batch_size, move(dataset_augmentations), ColorType::GRAY);

    // Allocate memory for x and y tensors
    cout << "Create x and y" << endl;
    tensor x = eddlT::create({ batch_size, d.n_channels_, d.resize_dims_[0], d.resize_dims_[1] });
    tensor y = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });

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

    return EXIT_SUCCESS;
}