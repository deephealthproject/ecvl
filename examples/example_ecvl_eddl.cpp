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
#include "ecvl/eddl.h"

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

    int batch_size = 64;
    cout << "Creating a DLDataset" << endl;
    DLDataset d("../examples/data/mnist/mnist.yml", batch_size, { 28,28 }, ColorType::GRAY);

    // Allocate memory for x_train and y_train tensors
    cout << "Create x_train and y_train" << endl;
    tensor x_train = eddlT::create({ batch_size, d.n_channels_, d.resize_dims_[0], d.resize_dims_[1] });
    tensor y_train = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });

    // Load a batch of d.batch_size_ images into x_train and corresponding labels in y_train
    // Images are resized to the dimensions specified in size
    cout << "Executing LoadBatch" << endl;
    d.LoadBatch(x_train, y_train);

    // Load all the split (e.g., Test) images in x_test and corresponding labels in y_test
    tensor x_test;
    tensor y_test;
    cout << "Executing TestToTensor" << endl;
    TestToTensor(d, d.resize_dims_, x_test, y_test, ColorType::GRAY);

    return EXIT_SUCCESS;
}