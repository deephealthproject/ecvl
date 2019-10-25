#include <iostream>
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    // Open a nifti image
    Image nifti_image;
    cout << "Reading a nifti Image" << endl;
    if (!ImRead("../data/nifti/LR_nifti.nii", nifti_image, ImageFormat::NIFTI)) {
        return EXIT_FAILURE;
    }

    // Apply some processing
    int gamma = 3;
    cout << "Executing GammaContrast" << endl;
    GammaContrast(nifti_image, nifti_image, gamma);

    // Save an Image in the nifti format
    cout << "Save a nifti Image" << endl;
    ImWrite("nifti_gamma.nii", nifti_image, ImageFormat::NIFTI);

    // Open a dicom image
    Image dicom_image;
    cout << "Reading a dicom Image" << endl;
    if (!ImRead("../data/isic_dicom/ISIC_0000008.dcm", dicom_image, ImageFormat::DICOM)) {
        return EXIT_FAILURE;
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
    ImWrite("dicom_thresholded.dcm", dicom_image, ImageFormat::DICOM);

    // Any Image can be saved in the nifti or dicom format
    Image img;
    if (!ImRead("../data/test.jpg", img)) {
        return EXIT_FAILURE;
    }
    Mirror2D(img, img);
    ChangeColorSpace(img, img, ColorType::RGB);

    cout << "Save a nifti from .jpg Image" << endl;
    ImWrite("nifti_mirror.nii", img, ImageFormat::NIFTI);
    cout << "Save a dicom from .jpg Image" << endl;
    ImWrite("dicom_mirror.dcm", img, ImageFormat::DICOM);

    return EXIT_SUCCESS;
}