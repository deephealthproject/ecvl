/*
* ECVL - European Computer Vision Library
* Version: 1.0.0
* copyright (c) 2021, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef SUPPORT_OPENSLIDE_H_
#define SUPPORT_OPENSLIDE_H_

#include "ecvl/core/filesystem.h"
#include "ecvl/core/image.h"
#include "openslide.h"

namespace ecvl
{
class OpenSlideImage
{
    openslide_t* osr_;
    const ecvl::filesystem::path filename_;
    int n_levels_;

public:
    OpenSlideImage(const ecvl::filesystem::path& filename) : filename_{ filename }
    {
        osr_ = openslide_open(filename.string().c_str());
        if (osr_ == NULL || openslide_get_error(osr_) != NULL) {
            ECVL_ERROR_CANNOT_LOAD_IMAGE;
        }
        n_levels_ = openslide_get_level_count(osr_);
    }

    /** @brief Get the number of levels in the image. */
    int GetLevelCount() { return n_levels_; }

    /** @brief Get width and height for each level of a whole-slide image.

    @param[out] levels A std::vector of array containing two elements, width and height respectively.
                levels[k] are the dimensions of level k.
    */
    void GetLevelsDimensions(std::vector<std::array<int, 2>>& levels);

    /** @brief Get the downsampling factor for each level, or -1.0 if an error occurred.

    @param[out] levels It contains the downsampling factor for the corresponding level of that position.
    */
    void GetLevelDownsamples(std::vector<double>& levels);

    /** @brief Get the best level to use for displaying the given downsample.

    @param[in] downsample The downsample desired factor.

    @return level OpenSlide image level extracted, or -1 if an error occurred.
    */
    int GetBestLevelForDownsample(const double& downsample);

    /** @brief Loads properties (metadata) from the OpenSlide file and saves them into an ECVL Image.

    @param[out] dst Image in which metadata will be stored.
    */
    void GetProperties(Image& dst);

    /** @brief Loads a region of a whole-slide image.

    Supported formats are those supported by OpenSlide library.
    If the region cannot be read for any reason, the function creates an empty Image and returns false.

    @anchor openslideread_path

    @param[out] dst Image in which data will be stored. It will be a RGB image stored in a "cxy" layout.
    @param[in] level OpenSlide image level to be extracted.
    @param[in] dims std::vector containing { x, y, w, h }.
                x and y are the top left x-coordinate and y-coordinate, in the level 0 reference frame.
                w and h are the width and height of the region.

    @return true if the region is correctly read, false otherwise.
    */
    bool ReadRegion(Image& dst, const int level, const std::vector<int>& dims);

    /** @brief Close the OpenSlide object. */
    void Close()
    {
        openslide_close(osr_);
        osr_ = nullptr;
    }

    ~OpenSlideImage()
    {
        if (osr_) {
            Close();
        }
    }
};

/** @example example_openslide.cpp
 Openslide support example.
*/
} // namespace ecvl

#endif // SUPPORT_OPENSLIDE_H_
