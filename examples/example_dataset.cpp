#include "ecvl/core.h"
#include "ecvl/dataset_parser.h"

#include <filesystem>
#include <iostream>

using std::cout;
using std::endl;

int main()
{
    std::filesystem::path file = "mnist/mnist.yml";
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