/*
* ECVL - European Computer Vision Library
* Version: 0.3.4
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

#include "ecvl/augmentations.h"
#include "ecvl/core.h"
#include "ecvl/support_eddl.h"
#include "ecvl/core/filesystem.h"

using namespace ecvl;
using namespace ecvl::filesystem;
using namespace eddl;
using namespace std;

layer LeNet(layer x, const int& num_classes);
layer VGG16(layer x, const int& num_classes);

int main()
{
    // Required for enabling nested parallelism
    omp_set_nested(1);

    // Create the augmentations to be applied to the dataset images during training and test.
    auto training_augs = make_shared<SequentialAugmentationContainer>(
        AugCenterCrop(),
        AugResizeDim({ 224,224 }),
        AugRotate({ -5, 5 }),
        AugAdditiveLaplaceNoise({ 0, 0.2 * 255 }),
        AugCoarseDropout({ 0, 0.55 }, { 0.02,0.1 }, 0),
        AugAdditivePoissonNoise({ 0, 40 }),
        AugToFloat32(255)
        );

    auto test_augs = make_shared<SequentialAugmentationContainer>(AugToFloat32(255));

    // Replace the random seed with a fixed one to have reproducible experiments
    AugmentationParam::SetSeed(0);

    DatasetAugmentations dataset_augmentations{ { training_augs, test_augs } };

    constexpr int batch_size = 8;
    constexpr double queue_ratio = 1.;
    unsigned num_workers = 4;

    cout << "Creating a DLDataset" << endl;
    //DLDataset d("../examples/data/mnist/mnist.yml", batch_size, dataset_augmentations, ColorType::GRAY, ColorType::none, num_workers, queue_ratio, { true, false });
    //DLDataset d("../examples/data/mnist/mnist_reduced.yml", batch_size, dataset_augmentations, ColorType::GRAY, ColorType::none, num_workers, queue_ratio, { false, false });
    DLDataset d("D:/dataset/isic_classification_2018/isic_classification_2018.yml", batch_size, dataset_augmentations, ColorType::RGB, ColorType::none, num_workers, queue_ratio, { true, false, false });
    ofstream of;
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;
    constexpr int epochs = 5;

    //layer in = Input({ 1,28,28 });
    //layer out = Softmax(LeNet(in, 10));
    layer in = Input({ 3,224,224 });
    layer out = Softmax(VGG16(in, 7));
    model net = Model({ in }, { out });

    // Build model
    build(net,
        sgd(0.01f, 0.9f),
        { "sce" },
        { "accuracy" },
        CS_GPU({ 1 }, "low_mem")
    );
    summary(net);

    auto num_batches_training = d.GetNumBatches(SplitType::training);
    auto num_batches_test = d.GetNumBatches(SplitType::test);

    vector<Sample> samples;
    shared_ptr<Tensor>x, y;

    for (int i = 0; i < epochs; ++i) {
        tm_epoch.reset();
        tm_epoch.start();
        // Resize to batch_size if we have done a resize previously
        if (d.split_[d.current_split_].last_batch_ != batch_size) {
            net->resize(batch_size);
        }

        cout << "Starting training" << endl;
        d.SetSplit(SplitType::training);

        // Reset current split with shuffling
        d.ResetBatch(d.current_split_, true);

        // Two threads: the consumer and the other is delegated to spawn the producers
        #pragma omp parallel num_threads(2)
        {
            const int thread_index = omp_get_thread_num();
            if (thread_index == 0) {
                // Create producers
                #pragma omp parallel num_threads(num_workers)
                {
                    const int prod_index = omp_get_thread_num();
                    d.ThreadFunc(prod_index);
                }
            }
            else {
                // Consumer thread
                for (int j = 0; j < num_batches_training; ++j) {
                    tm.reset();
                    tm.start();
                    cout << "Epoch " << i << "/" << epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
                    cout << "|fifo| " << d.GetQueueSize() << " - ";

                    tie(samples, x, y) = d.GetBatch();

                    // if it's the last batch and the number of samples doesn't fit the batch size, resize the network
                    if (j == num_batches_training - 1 && x->shape[0] != batch_size) {
                        net->resize(x->shape[0]);
                    }

                    train_batch(net, { x.get() }, { y.get() });
                    print_loss(net, j);
                    tm.stop();
                    cout << "Elapsed time: " << tm.getTimeMilli() << " ms" << endl;
                }
            }
        }

        cout << "Starting test" << endl;
        d.SetSplit(SplitType::test);

        // Reset current split without shuffling
        d.ResetBatch(d.current_split_, false);
        #pragma omp parallel num_threads(2)
        {
            const int thread_index = omp_get_thread_num();
            if (thread_index == 0) {
                // Create producers
                #pragma omp parallel num_threads(num_workers)
                {
                    const int prod_index = omp_get_thread_num();
                    cout << prod_index << endl;
                    d.ThreadFunc(prod_index);
                }
            }
            else {
                // Consumer thread

                for (int j = 0; j < num_batches_test; ++j) {
                    tm.reset();
                    tm.start();
                    cout << "Test: Epoch " << i << "/" << epochs - 1 << " (batch " << j << "/" << num_batches_test - 1 << ") - ";
                    cout << "|fifo| " << d.GetQueueSize() << " - ";

                    tie(samples, x, y) = d.GetBatch();

                    // Resize net for last batch
                    if (auto x_batch = x->shape[0]; j == num_batches_test - 1 && x_batch != batch_size) {
                        // last mini-batch could have different size
                        net->resize(x_batch);
                    }
                    eval_batch(net, { x.get() }, { y.get() });
                    print_loss(net, j);

                    tm.stop();
                    cout << "Elapsed time: " << tm.getTimeMilli() << " ms" << endl;
                }
            }
        }

        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
    }

    return EXIT_SUCCESS;
}
layer LeNet(layer x, const int& num_classes)
{
    x = MaxPool(ReLu(Conv(x, 20, { 5,5 })), { 2,2 }, { 2,2 });
    x = MaxPool(ReLu(Conv(x, 50, { 5,5 })), { 2,2 }, { 2,2 });
    x = Reshape(x, { -1 });
    x = ReLu(Dense(x, 500));
    x = Dense(x, num_classes);

    return x;
}
layer VGG16(layer x, const int& num_classes)
{
    x = ReLu(Conv(x, 64, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 64, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 128, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 128, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 256, { 3,3 }));
    x = ReLu(Conv(x, 256, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 256, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 512, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 512, { 3,3 })), { 2,2 }, { 2,2 });

    x = Reshape(x, { -1 });
    x = ReLu(Dense(x, 4096));
    x = ReLu(Dense(x, 4096));
    x = Dense(x, num_classes);

    return x;
}