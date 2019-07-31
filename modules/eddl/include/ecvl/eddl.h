#ifndef ECVL_EDDL_H_
#define ECVL_EDDL_H_

#include <eddl/apis/eddl.h>
#include "ecvl/core/image.h"

namespace ecvl {
    /** @brief Convert an EDDL Tensor into an ECVL Image.

    Tensor dimensions must be \f$C\f$ x \f$Y\f$ x \f$X\f$ or \f$Z\f$ x \f$C\f$ x \f$Y\f$ x \f$X\f$, where: \n
    \f$Z\f$ = depth \n
    \f$C\f$ = color channels \n
    \f$Y\f$ = height \n
    \f$X\f$ = width

    @param[in] t Input EDDL Tensor.
    @param[in] c_type ecvl::ColorType of input data (optional). \n
    If c_type is ColorType::none (default), it is assumed that: \n
    If the input has 4 channels, the color type is assumed to be ColorType::RGBA. \n
    If the input has 3 channels, the color type is assumed to be ColorType::BGR. \n
    If the input has 1 channels, the color type is assumed to be ColorType::GRAY. \n
    In any other case, the color type is assumed to be ColorType::none.

    @return ECVL Image.
    */
    Image TensorToImage(tensor& t, ColorType c_type = ColorType::none);

    /** @brief Convert an EDDL Tensor into an ECVL View.

    Tensor dimensions must be \f$C\f$ x \f$Y\f$ x \f$X\f$ or \f$Z\f$ x \f$C\f$ x \f$Y\f$ x \f$X\f$, where: \n
    \f$Z\f$ = depth \n
    \f$C\f$ = color channels \n
    \f$Y\f$ = height \n
    \f$X\f$ = width

    @param[in] t Input EDDL Tensor.
    @param[in] c_type ecvl::ColorType of input data (optional). \n
    If c_type is ColorType::none (default), it is assumed that: \n
    If the input has 4 channels, the color type is assumed to be ColorType::RGBA. \n
    If the input has 3 channels, the color type is assumed to be ColorType::BGR. \n
    If the input has 1 channels, the color type is assumed to be ColorType::GRAY. \n
    In any other case, the color type is assumed to be ColorType::none.

    @return ECVL View.
    */
    View<DataType::float32> TensorToView(tensor& t, ColorType c_type = ColorType::none);

    /** @brief Convert an ECVL Image into an EDDL Tensor.

    Image must have 3 or 4 dimensions. \n
    If the Image has 3 dimensions, the output Tensor will be created with shape \f$C\f$ x \f$Y\f$ x \f$X\f$. \n
    If the Image has 4 dimensions, the output Tensor will be created with shape \f$Z\f$ x \f$C\f$ x \f$Y\f$ x \f$X\f$.

    @param[in] img Input ECVL Image.

    @return EDDL Tensor.
    */
    tensor ImageToTensor(const Image& img);

    /** @brief Convert a set of images into a single EDDL Tensor.

    @param[in] dataset Vector of all images path.
    @param[in] dims Dimensions of the dataset in the form: number of total images, number of color channels of the images,
    and the width and height at which all the images has to be resized. Output Tensor will have these dimensions. \n
    \f$\{number\_of\_samples, number\_of\_colors\_channels, width, height\}\f$

    @return EDDL Tensor.
    */
    tensor DatasetToTensor(vector<string> dataset, const std::vector<int>& dims);

} // namespace ecvl 

#endif // ECVL_EDDL_H_