#ifndef ECVL_EDDLL_H_
#define ECVL_EDDLL_H_

#include <eddll/apis/eddl.h>

#include "ecvl/core/image.h"

namespace ecvl {

    /** @brief Convert a EDDLL Tensor into an ECVL Image.

    @param[in] t Input EDDLL Tensor.

    @return ECVL Image.
    */
    Image TensorToImage(tensor& t);

    /** @brief Convert an ECVL Image into EDDLL Tensor.

    @param[in] img Input ECVL Image.

    @return EDDLL Tensor.
    */
    tensor ImageToTensor(const Image& img);

    /** @brief Convert a set of images into a single EDDLL Tensor.

    @param[in] dataset Vector of all images path.
    @param[in] dims Dimensions of the dataset in the form: number of total images, number of channels of the images, 
    and the width and height at which all the images has to be resized. \f$\{number\_of\_samples, number\_of\_channels, width, height\}\f$

    @return EDDLL Tensor.
    */

    tensor DatasetToTensor(vector<string> dataset, const std::vector<int>& dims);
} // namespace ecvl 

#endif // ECVL_EDDLL_H_