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
    if (!ImRead("../data/test.jpg", img)) {
        return EXIT_FAILURE;
    }

    // Convert an Image into tensor
    cout << "Executing ImageToTensor" << endl;
    tensor t = ImageToTensor(img);

    // Apply eddl functions
    eddlT::div(t, 128);

    // Convert a tensor into an Image
    cout << "Executing TensorToImage" << endl;
    img = TensorToImage(t);

    // Convert a tensor into a View (they point to the same data)
    View<DataType::float32> view;
    cout << "Executing TensorToView" << endl;
    view = TensorToView(t);

    int batch_size = 64;
    cout << "Creating a DLDataset" << endl;
    DLDataset d("mnist/mnist.yml", batch_size, "training", ColorType::GRAY);

    // Allocate memory for x_train and y_train tensors
    std::vector<int> size{ 28,28 };
    cout << "Create x_train and y_train" << endl;
    tensor x_train = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y_train = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });

    // Load a batch of d.batch_size_ images into x_train and corresponding labels in y_train
    // Images are resized to the dimensions specified in size
    cout << "Executing LoadBatch" << endl;
    d.LoadBatch(size, x_train, y_train);

    // Load all the split (e.g., Test) images in x_test and corresponding labels in y_test
    tensor x_test;
    tensor y_test;
    cout << "Executing TestToTensor" << endl;
    TestToTensor(d, size, x_test, y_test, ColorType::GRAY);

    return EXIT_SUCCESS;
}