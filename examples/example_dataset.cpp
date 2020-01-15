#include <iostream>
#include "ecvl/core.h"
#include "ecvl/dataset_parser.h"

using std::cout;
using std::endl;

int main()
{
    auto path = "mnist/mnist.yml";
    cout << "Reading Dataset from '" << path << "' file" << endl;
    ecvl::Dataset d(path);

    cout << "Dataset name: '" << d.name_ << "'." << endl;
    cout << "Dataset description: '" << d.description_ << "'." << endl;
    cout << "Dataset classes: <";
    for (auto& i : d.classes_) {
        cout << i << ",";
    }
    cout << ">" << endl;

    return EXIT_SUCCESS;
}