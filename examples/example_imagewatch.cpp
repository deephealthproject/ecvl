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

using namespace ecvl;
using namespace std;

// This is not a real snippet but contains a bunch of images with 
// different types to test the ImageWatch(natvis) functionalities

int main()
{
    Image a({ 5, 5, 1 }, DataType::uint8, "xyc", ColorType::GRAY);
    Image b({ 5, 5, 3 }, DataType::uint8, "xyc", ColorType::RGB);
    Image c({ 5, 5, 3 }, DataType::uint8, "xyc", ColorType::BGR);

    Image d({ 1, 5, 5 }, DataType::uint8, "cxy", ColorType::GRAY);
    Image e({ 3, 5, 5 }, DataType::uint8, "cxy", ColorType::RGB);
    Image f({ 3, 5, 5 }, DataType::uint8, "cxy", ColorType::BGR);

    Image g({ 5, 5 }, DataType::uint8, "xy", ColorType::none);
    Image h({ 5, 5, 2 }, DataType::uint8, "xyz", ColorType::none);
    Image i({ 5, 5, 3 }, DataType::uint8, "xyz", ColorType::none);
    Image j({ 5, 5, 4 }, DataType::uint8, "xyz", ColorType::none);

    return EXIT_SUCCESS;
}