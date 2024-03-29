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
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    // Open an Image
    Image img1, img2, tmp;
    if (!ImRead("../examples/data/test.jpg", img1)) {
        return EXIT_FAILURE;
    }

    // Resize an Image to new_width, new_height (optional: InterpolationType)
    int new_width = 225;
    int new_height = 300;
    cout << "Executing ResizeDim" << endl;
    ResizeDim(img1, tmp, { new_width, new_height }, InterpolationType::nearest);
    ImWrite("img_resized.jpg", tmp);

    // Resize an Image by scaling the dimensions to a given scale factor res_scale (optional: InterpolationType)
    vector<double> res_scale = { 2,2 };
    cout << "Executing ResizeScale" << endl;
    ResizeScale(img1, tmp, res_scale, InterpolationType::cubic);
    ImWrite("img_resized_scale.jpg", tmp);

    // Flip an Image
    cout << "Executing Flip2D" << endl;
    Flip2D(img1, tmp);
    ImWrite("img_flipped.jpg", tmp);

    // Mirror an Image
    cout << "Executing Mirror2D" << endl;
    Mirror2D(img1, tmp);
    ImWrite("img_mirrored.jpg", tmp);

    // Transpose an Image
    cout << "Executing Transpose" << endl;
    Transpose(img1, tmp);
    ImWrite("img_transpose.jpg", tmp);

    // Rotate an Image of a given angle (optional: coordinates of the rotation center, scaling factor and InterpolationType)
    int angle = 60;
    cout << "Executing Rotate2D" << endl;
    Rotate2D(img1, tmp, angle);
    ImWrite("img_rotated.jpg", tmp);

    // Rotate an Image of a given angle; it is scaled during rotation with a rot_scale scaling factor. The output is resized accordingly.
    double rot_scale = 1.5;
    cout << "Executing RotateFullImage2D" << endl;
    RotateFullImage2D(img1, tmp, angle, rot_scale);
    ImWrite("img_rotated_full.jpg", tmp);

    // Change the color space of an Image from BGR to GRAY
    cout << "Executing ChangeColorSpace" << endl;
    ChangeColorSpace(img1, tmp, ColorType::GRAY);
    ImWrite("img_gray.jpg", tmp);

    // Calculate the Otsu thresholding value (the Image must be GRAY)
    cout << "Executing OtsuThreshold" << endl;
    double thresh = OtsuThreshold(tmp);

    //Apply a fixed threshold to an input Image (optional: ThresholdingType)
    double maxval = 255;
    cout << "Executing Threshold" << endl;
    Threshold(tmp, tmp, thresh, maxval);
    ImWrite("img_thresh.png", tmp);

    // Label connected components in a binary Image
    Image labels;
    ConnectedComponentsLabeling(tmp, labels);
    ImWrite("img_labels.png", labels);

    // Find contours in a binary Image
    vector<vector<ecvl::Point2i>> contours;
    FindContours(tmp, contours);

    // Create and populate a kernel Image. Kernel must be float64, "xyc" and with one color channel
    Image kernel({ 3, 3, 1 }, DataType::float64, "xyc", ColorType::GRAY);
    auto i = kernel.Begin<double>(), e = kernel.End<double>();
    float c = 0.11f;
    for (; i != e; ++i) {
        *i = c;
    }

    // Convolves an Image with a kernel (optional: destination DataType)
    cout << "Executing Filter2D" << endl;
    Filter2D(img1, tmp, kernel);
    ImWrite("img_filter.jpg", tmp);

    // Convolves an Image with a couple of 1-dimensional kernels (optional: destination DataType)
    vector<double> kernelX = { 1, 2, 1 };
    vector<double> kernelY = { 1, 0, -1 };
    cout << "Executing SeparableFilter2D" << endl;
    SeparableFilter2D(img1, tmp, kernelX, kernelY);
    ImWrite("img_separable_filter.jpg", tmp);

    // Blur an Image using a Gaussian kernel (optional: standard deviation in Y direction)
    cout << "Executing GaussianBlur" << endl;
    GaussianBlur(img1, tmp, 5, 5, 0.0);
    ImWrite("img_gaussian_blur.jpg", tmp);

    // Add Laplace distributed noise to an Image
    float stddev = 255 * 0.05;
    cout << "Executing AdditiveLaplaceNoise" << endl;
    AdditiveLaplaceNoise(img1, tmp, stddev);
    ImWrite("img_laplacenoise.jpg", tmp);

    // Adjust contrast by scaling each pixel value X to 255 * ((X/255) ** gamma)
    int gamma = 3;
    cout << "Executing GammaContrast" << endl;
    GammaContrast(img1, tmp, gamma);
    ImWrite("img_gammacontrast.jpg", tmp);

    // Set rectangular areas within an Image to zero
    float prob = 0.5;
    float drop_size = 0.1f;
    bool per_channel = true;
    cout << "Executing CoarseDropout" << endl;
    CoarseDropout(img1, tmp, prob, drop_size, per_channel);
    ImWrite("img_coarsedropout.jpg", tmp);

    //Horizontal concatenation of images
    vector<Image> images;
    if (!ImRead("../examples/data/img0003.png", img1)) {
        return EXIT_FAILURE;
    }
    if (!ImRead("../examples/data/img0015.png", img2)) {
        return EXIT_FAILURE;
    }
    images.push_back(img1);
    images.push_back(img2);
    images.push_back(img1);
    images.push_back(img2);

    ResizeDim(images[1], images[1], { images[1].dims_[0] / 2, images[1].dims_[1] });
    cout << "Executing HConcat" << endl;
    HConcat(images, tmp);
    ImWrite("img_hconcat.jpg", tmp);

    images.erase(images.begin() + 1);
    // Vertical concatenation of different images
    ResizeDim(images[1], images[1], { images[1].dims_[0] , images[1].dims_[1] / 2 });
    cout << "Executing VConcat" << endl;
    VConcat(images, tmp);
    ImWrite("img_vconcat.jpg", tmp);
    images.erase(images.begin() + 1);

    // Stack a sequence of Images along the channel dimension (xyc -> xyo)
    cout << "Executing Stack of images xyc creating a xyo Image" << endl;
    Stack(images, tmp);

    cout << "Executing Salt" << endl;
    Salt(img1, tmp, 0.5, true);
    ImWrite("img_salt_perchannel.png", tmp);
    Salt(img1, tmp, 0.5, false);
    ImWrite("img_salt.png", tmp);

    cout << "Executing Pepper" << endl;
    Pepper(img1, tmp, 0.5, true);
    ImWrite("img_pepper_perchannel.png", tmp);
    Pepper(img1, tmp, 0.5, false);
    ImWrite("img_pepper.png", tmp);

    cout << "Executing SaltAndPepper" << endl;
    SaltAndPepper(img1, tmp, 0.5, true);
    ImWrite("img_saltandpepper_perchannel.png", tmp);
    SaltAndPepper(img1, tmp, 0.5, false);
    ImWrite("img_saltandpepper.png", tmp);

    cout << "Executing OpticalDistortion" << endl;
    OpticalDistortion(img1, tmp);
    ImWrite("img_opticaldistortion.png", tmp);

    cout << "Executing GridDistortion" << endl;
    GridDistortion(img1, tmp);
    ImWrite("img_griddistortion.png", tmp);

    cout << "Executing ElasticTransform" << endl;
    ElasticTransform(img1, tmp);
    ImWrite("img_elastictransform.png", tmp);

    cout << "Executing CenterCrop" << endl;
    CenterCrop(img1, tmp, { 225, 300 });
    ImWrite("img_centercrop.png", tmp);


    return EXIT_SUCCESS;
}