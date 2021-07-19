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

void Run(DLDataset& d) {
    ofstream of;
    cv::TickMeter tm;
    cv::TickMeter tm_epoch;
    constexpr int epochs = 1;

    auto num_batches_training = d.GetNumBatches(SplitType::training);
    auto num_batches_test = d.GetNumBatches(SplitType::test);

    for (int i = 0; i < epochs; ++i) {
        tm_epoch.reset();
        tm_epoch.start();
        /* Resize to batch_size if we have done a resize previously
        if (d.split_[d.current_split_].last_batch_ != batch_size){
            net->resize(batch_size);
        }
        */
        cout << "Starting training" << endl;
        d.SetSplit(SplitType::training);

        // Reset current split with shuffling
        d.ResetBatch(d.current_split_, true);

        // Spawn num_workers threads
        d.Start();
        for (int j = 0; j < num_batches_training; ++j) {
            tm.reset();
            tm.start();
            cout << "Epoch " << i << "/" << epochs - 1 << " (batch " << j << "/" << num_batches_training - 1 << ") - ";
            cout << "|fifo| " << d.GetQueueSize() << " - ";

            // tuple<vector<Sample>, shared_ptr<Tensor>, shared_ptr<Tensor>> samples_and_labels;
            // samples_and_labels = d.GetBatch();
            // or...
            auto [samples, x, y] = d.GetBatch();

            // Sleep in order to simulate EDDL train_batch
            cout << "sleeping...";
            this_thread::sleep_for(chrono::milliseconds(50));
            // eddl::train_batch(net, { x.get() }, { y.get() });

            tm.stop();
            cout << "Elapsed time: " << tm.getTimeMilli() << endl;
        }
        d.Stop();

        cout << "Starting test" << endl;
        d.SetSplit(SplitType::test);

        // Reset current split without shuffling
        d.ResetBatch(d.current_split_, false);

        d.Start();
        for (int j = 0; j < num_batches_test; ++j) {
            tm.reset();
            tm.start();
            cout << "Test: Epoch " << i << "/" << epochs - 1 << " (batch " << j << "/" << num_batches_test - 1 << ") - ";
            cout << "|fifo| " << d.GetQueueSize() << " - ";

            // tuple<vector<Sample>, shared_ptr<Tensor>, shared_ptr<Tensor>> samples_and_labels;
            // samples_and_labels = d.GetBatch();
            // or...
            auto [_, x, y] = d.GetBatch();

            /* Resize net for last batch
            if (auto x_batch = x->shape[0]; j == num_batches_test - 1 && x_batch != batch_size) {
                // last mini-batch could have different size
                net->resize(x_batch);
            }
            */
            // Sleep in order to simulate EDDL evaluate_batch
            cout << "sleeping... - ";
            this_thread::sleep_for(chrono::milliseconds(50));
            // eddl::eval_batch(net, { x.get() }, { y.get() });

            tm.stop();
            cout << "Elapsed time: " << tm.getTimeMilli() << endl;
        }
        d.Stop();
        tm_epoch.stop();
        cout << "Epoch elapsed time: " << tm_epoch.getTimeSec() << endl;
    }
}

int main()
{
    // Create the augmentations to be applied to the dataset images during training and test.
    auto training_augs = make_shared<SequentialAugmentationContainer>(
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

    constexpr int batch_size = 500;
    constexpr double queue_ratio = 20.;
    unsigned num_workers = 1;

    cout << "Running tests with " << num_workers << endl;
    cout << "Creating a DLDataset" << endl;
    DLDataset d("../examples/data/mnist/mnist_reduced.yml", batch_size, dataset_augmentations, ColorType::GRAY, ColorType::none, num_workers, queue_ratio, { false, false });
    //DLDataset d("D:/Data/isic_skin_lesion/isic_skin_lesion/isic_classification.yml", batch_size, dataset_augmentations, ColorType::RGB, ColorType::none, num_workers, queue_ratio);
    Run(d);

    num_workers = 2u;
    cout << "Running tests with " << num_workers << endl;
    d.SetWorkers(num_workers);
    Run(d);

    num_workers = 8u;
    cout << "Running tests with " << num_workers << endl;
    d.SetWorkers(num_workers);
    Run(d);

    return EXIT_SUCCESS;
}