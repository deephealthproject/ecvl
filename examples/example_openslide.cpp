#include <iostream>
#include "ecvl/core.h"

using namespace ecvl;
using namespace std;

int main()
{
    Image img;

    // Image level to be extracted
    int level = 0;
    //Set the RegionOfInterest informations
    vector<int> dims = {
        11978, // x
        30243, // y
        3341,  // w
        3797   // h
    };

    // Read an Hamamatsu file
    if (!HamamatsuRead("../data/hamamatsu/10-B1-TALG.ndpi", img, level, dims)) {
        return EXIT_FAILURE;
    }
    ImWrite("hamamatsu_1.png", img);
    cout << "Writing 'hamamatsu_1.png'\n";

    dims = {
        3386,  // x
        36837, // y
        3355,  // w
        4447   // h
    };
    if (!HamamatsuRead("../data/hamamatsu/11-B1TALG.ndpi", img, level, dims)) {
        return EXIT_FAILURE;
    }
    ImWrite("hamamatsu_2.png", img);
    cout << "Writing 'hamamatsu_2.png'\n";

    return EXIT_SUCCESS;
}