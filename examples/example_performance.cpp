/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core.h"
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>

using namespace ecvl;
using namespace std;


struct Benchmark
{
    int n_test_ = 1000;
    cv::TickMeter tm_;
    vector<double> timings_;

    Benchmark() {}
    Benchmark(int n_test) : n_test_(n_test) {}

    template <class Functor, class... Args>
    double Run(int processor_count, Functor f, Args&&... args)
    {
        omp_set_num_threads(processor_count);
        for (int i = 0; i < n_test_; ++i) {
            tm_.reset();
            tm_.start();
            f(args...);
            tm_.stop();
            timings_.push_back(tm_.getTimeMilli());
        }
        return std::accumulate(timings_.begin(), timings_.end(), 0.) / timings_.size();
    }
};

int main()
{
    // Open an Image
    Image in, out;
    ImRead("../examples/data/test.jpg", in, ImReadMode::GRAYSCALE);

    Benchmark b;
    auto processor_count = std::thread::hardware_concurrency();
    if (!processor_count) {
        return EXIT_FAILURE;
    }

    cout << "CPU Cores: " << processor_count << endl << endl;

    cout << "Benchmarking Threshold" << endl;
    auto time = b.Run(1, Threshold, in, out, 128, 255., ThresholdingType::BINARY);
    cout << "Elapsed sequential: " << time << endl;

    time = b.Run(processor_count, Threshold, in, out, 128, 255., ThresholdingType::BINARY);
    cout << "Elapsed parallel: " << time << endl << endl;

    cout << "Benchmarking Mirror2D" << endl;
    time = b.Run(1, Mirror2D, in, out);
    cout << "Elapsed sequential: " << time << endl;

    time = b.Run(processor_count, Mirror2D, in, out);
    cout << "Elapsed parallel: " << time << endl << endl;

    cout << "Benchmarking Flip2D" << endl;
    time = b.Run(1, Flip2D, in, out);
    cout << "Elapsed sequential: " << time << endl;

    time = b.Run(processor_count, Flip2D, in, out);
    cout << "Elapsed parallel: " << time << endl << endl;

    //Image ccl_in;
    //Threshold(in, ccl_in, 128, 255);
    //cout << "Benchmarking ConnectedComponentsLabeling" << endl;
    //time = b.Run(1, ConnectedComponentsLabeling, ccl_in, out);
    //cout << "Elapsed sequential: " << time << endl;

    //time = b.Run(processor_count, ConnectedComponentsLabeling, ccl_in, out);
    //cout << "Elapsed parallel: " << time << endl << endl;

    //cout << "Benchmarking HConcat" << endl;
    //vector images{ in,in };
    //time = b.Run(1, HConcat, images, out);
    //cout << "Elapsed sequential: " << time << endl;

    //time = b.Run(processor_count, HConcat, images, out);
    //cout << "Elapsed parallel: " << time << endl << endl;

    return EXIT_SUCCESS;
}