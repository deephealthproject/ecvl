#include <iostream>
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    Image img;

    //Set the RegionOfInterest informations
    int x = 11978, y = 30243;
    int w = 3341, h = 3797;

    //Read an Hamamatsu file
    if (!HamamatsuRead("D:/Data/HAMAMATSU/UC2_Dev_ExampleFiles/10-B1-TALG.ndpi", img, x, y, w, h)) {
        return EXIT_FAILURE;
    }
    ImWrite("hamamatsu_1.png", img);

    x = 3386, y = 36837;
    w = 3355, h = 4447;
    if (!HamamatsuRead("D:/Data/HAMAMATSU/UC2_Dev_ExampleFiles/11-B1TALG.ndpi", img, x, y, w, h)) {
        return EXIT_FAILURE;
    }
    ImWrite("hamamatsu_2.png", img);

    return EXIT_SUCCESS;
}