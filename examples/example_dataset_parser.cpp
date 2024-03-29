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

#include "ecvl/core.h"
#include "ecvl/core/filesystem.h"
#include "ecvl/dataset_parser.h"

#include <iostream>

using ecvl::filesystem::path;
using std::cout;
using std::endl;

int main()
{
    path file = "../examples/data/mnist/mnist.yml";
    cout << "Reading Dataset from " << file << " file" << endl;
    ecvl::Dataset d(file);

    cout << "Dataset name: '" << d.name_ << "'." << endl;
    cout << "Dataset description: '" << d.description_ << "'." << endl;
    cout << "Dataset classes: <";
    for (auto& i : d.classes_) {
        cout << i << ",";
    }
    cout << ">" << endl;

    return EXIT_SUCCESS;
}