/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
* copyright (c) 2020, UniversitÓ degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#define _USE_MATH_DEFINES

#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    // Loads an existing image
    Image img;
    if (!ImRead("../examples/data/example_ISIC_01.jpg", img, ImReadMode::COLOR)) {
        return EXIT_FAILURE;
    }
    Image origin(img);
    ChangeColorSpace(img, img, ColorType::GRAY);

    // Threshold
    Threshold(img, img, 25, 1);

    // Calculate raw moments up to order 2 ... 
    Image moments;
    Moments(img, moments, 2);

    // ... and use them to calculate mass/area and center of mass {x, y} of the image objects
    ContiguousView<DataType::float64> rm(moments);
    double M00 = rm({ 0, 0 });
    double M10 = rm({ 1, 0 });
    double M01 = rm({ 0, 1 });
    double M11 = rm({ 1, 1 });
    double M02 = rm({ 0, 2 });
    double M20 = rm({ 2, 0 });
    
    // Centroid coordinates can be then calculated as
    double x = M10 / M00;
    double y = M01 / M00;

    // Now we can calculate the central moments ...
    // ... by using ECVL function
    CentralMoments(img, moments, {x, y});
    ContiguousView<DataType::float64> cm(moments);
    double u00 = cm({ 0, 0 });
    double u11 = cm({ 1, 1 });
    double u20 = cm({ 2, 0 });
    double u02 = cm({ 0, 2 });

    // ... or through the raw moments
    double u00_bis = M00;
    double u11_bis = M11 - x * M01;
    double u20_bis = M20 - x * M10;
    double u02_bis = M02 - y * M01;

    // Terms of the covariance matrix are thus
    double u_20 = u20 / u00; // or M20 / M00 - (x * x);
    double u_02 = u02 / u00; // or M02 / M00 - (y * y);
    double u_11 = u11 / u00; // or M11 / M00 - (x * y);

    // Eigenvalues (lambda1, labda2) and the orientation of the eigenvector can 
    // be calculated from the covariance matrix values as: 
    double lambda1 = (u_20 + u_02) / 2 + sqrt(4 * u_11*u_11 + (u_20 - u_02)*(u_20 - u_02)) / 2;
    double lambda2 = (u_20 + u_02) / 2 - sqrt(4 * u_11*u_11 + (u_20 - u_02)*(u_20 - u_02)) / 2;

    double theta = 0.5 * atan2(2 * u_11, (u_20 - u_02)); // rad

    // Eigenvalues are proportional to the square length of the eigenvector axes. So the half-axes
    // of the ellipse generated by the eigenvectors are given by d/sqrt(lambda1) and d/sqrt(lambda2)
    // where d is the proportional factor. Considering that the moment M00 is the area (the image on
    // which it is calculated is binary!) of the image objects we can easily calculate d
    double d = sqrt(M00 * sqrt(lambda1 * lambda2) / M_PI);

    // Hal-axes (a and b) are then
    double a = d / sqrt(lambda1);
    double b = d / sqrt(lambda2);

    // We can now draw the ellipses which has the same moments of the image objects
    DrawEllipse(origin, { (int)x, (int)y }, { (int)a, (int)b }, theta * 180 / M_PI, { 0, 0, 255 }, 2);
 
    return EXIT_SUCCESS;
}