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
    // Open a nifti image
    Image nifti_image;
    cout << "Reading a nifti Image" << endl;
    if (!NiftiRead("../examples/data/nifti/LR_nifti.nii", nifti_image)) {
        return EXIT_FAILURE;
    }

    // Resize all the dimensions
    ResizeDim(nifti_image, nifti_image, { 256, 256, 256 });

    // Apply some processing
    int gamma = 3;
    cout << "Executing GammaContrast" << endl;
    GammaContrast(nifti_image, nifti_image, gamma);

    // Save an Image in the nifti format
    cout << "Save a nifti Image" << endl;
    NiftiWrite("nifti_gamma.nii", nifti_image);

    // Open a dicom image
    Image dicom_image;
    cout << "Reading a dicom Image" << endl;
    if (!DicomRead("../examples/data/dicom/ISIC_0000008.dcm", dicom_image)) {
        return EXIT_FAILURE;
    }

    string key = "fake_meta";
    int value = 8;
    if (dicom_image.SetMeta(key, value)) {
        // insert a new metadata
        cout << "[" << key << "]" << " = " << value << " inserted" << endl;
    }
    else {
        // the key is already present
        cout << "[" << key << "]" << " = " << value << " updated" << endl;
    }

    // User searches for a specific metadata
    auto rows = dicom_image.GetMeta("Rows");
    // User knows what data type the metadata is
    unsigned short r = any_cast<unsigned short>(rows.Get());
    cout << "Rows: " << r << endl;
    // User doesn't know what data type the metadata is. This call will return a string containing the metadata value if present otherwise an empty string.
    string r_str = rows.GetStr();
    cout << "Rows: " << r_str << endl;

    // Loop over all the metadata and print them
    for (auto& p : dicom_image.meta_) {
        cout << p.first << " - " << p.second.GetStr() << endl;
    }

    // Apply some processing
    cout << "Executing ChangeColorSpace" << endl;
    ChangeColorSpace(dicom_image, dicom_image, ColorType::GRAY);
    cout << "Executing OtsuThreshold" << endl;
    double thresh = OtsuThreshold(dicom_image);
    double maxval = 255;
    cout << "Executing Threshold" << endl;
    Threshold(dicom_image, dicom_image, thresh, maxval);

    // Save an Image in the dicom format
    cout << "Save a dicom Image" << endl;
    DicomWrite("dicom_thresholded.dcm", dicom_image);

    // Any Image can be saved in the nifti or dicom format
    Image img;
    if (!ImRead("../examples/data/test.jpg", img)) {
        return EXIT_FAILURE;
    }
    Mirror2D(img, img);
    ChangeColorSpace(img, img, ColorType::RGB);

    cout << "Save a nifti from .jpg Image" << endl;
    NiftiWrite("nifti_mirror.nii", img);
    cout << "Save a dicom from .jpg Image" << endl;
    DicomWrite("dicom_mirror.dcm", img);

    return EXIT_SUCCESS;
}