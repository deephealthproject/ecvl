#ifndef ECVL_EDDL_H_
#define ECVL_EDDL_H_

#include <eddl/apis/eddl.h>
#include "ecvl/core/image.h"

namespace ecvl {

    /** @brief Convert a EDDL Tensor into an ECVL Image.

    @param[in] t Input EDDL Tensor.

    @return ECVL Image.
    */
    Image TensorToImage(tensor& t);

    /** @brief Convert an ECVL Image into EDDL Tensor.

    @param[in] img Input ECVL Image.

    @return EDDL Tensor.
    */
    tensor ImageToTensor(const Image& img);

    /** @brief Convert a set of images into a single EDDL Tensor.

    @param[in] dataset Vector of all images path.
    @param[in] dims Dimensions of the dataset in the form: number of total images, number of channels of the images, 
    and the width and height at which all the images has to be resized. \f$\{number\_of\_samples, number\_of\_channels, width, height\}\f$

    @return EDDL Tensor.
    */
    typedef void(*AugFunctions)(Image&);
    tensor DatasetToTensor(vector<string> dataset, const std::vector<int>& dims, vector<AugFunctions>aug_f = {});

} // namespace ecvl 

#endif // ECVL_EDDL_H_