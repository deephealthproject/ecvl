#include "../src/core.h"

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

    Image g({ 5, 5, 3 }, DataType::uint8, "xyz", ColorType::none);
    Image h({ 5, 5 }, DataType::uint8, "xy", ColorType::none);
    
    return EXIT_SUCCESS;
}