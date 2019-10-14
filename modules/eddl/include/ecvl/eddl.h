#ifndef ECVL_EDDL_H_
#define ECVL_EDDL_H_

#include <eddl/apis/eddl.h>
#include "ecvl/core/image.h"
#include "ecvl/dataset_parser.h"
#include <filesystem>

namespace ecvl {

class DLDataset : public Dataset {
public:
    int batch_size_;
    int current_batch_ = 0;
    int n_channels_;
    ColorType ctype_;
    std::string split_str_;

    DLDataset(const std::filesystem::path& filename, int batch_size, std::string split, ColorType ctype = ColorType::BGR) :
        Dataset{ filename },
        batch_size_{ batch_size },
        ctype_{ ctype },
        split_str_{ split },
        n_channels_{ this->samples_[0].LoadImage(ctype).Channels() }{}

    std::vector<int>& GetSplit();
    void SetSplit(const std::string& split_str);
};


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

/** @brief Load the training split of a Dataset (images and labels) into EDDL tensors.

@param[in] dataset Dataset object listing all the samples.
@param[in] size Dimensions (width and height) at which all the images have to be resized.
@param[out] images Tensor which contains all the images.
@param[out] labels Tensor which contains all the labels.

*/
void TrainingToTensor(const Dataset& dataset, const std::vector<int>& size, tensor& images, tensor& labels, ColorType ctype = ColorType::BGR);

/** @brief Load the validation split of a Dataset (images and labels) into EDDL tensors.

@param[in] dataset Dataset object listing all the samples.
@param[in] size Dimensions (width and height) at which all the images have to be resized.
@param[out] images Tensor which contains all the images.
@param[out] labels Tensor which contains all the labels.

*/
void ValidationToTensor(const Dataset& dataset, const std::vector<int>& size, tensor& images, tensor& labels, ColorType ctype = ColorType::BGR);

/** @brief Load the test split of a Dataset (images and labels) into EDDL tensors.

@param[in] dataset Dataset object listing all the samples.
@param[in] size Dimensions (width and height) at which all the images have to be resized.
@param[out] images Tensor which contains all the images.
@param[out] labels Tensor which contains all the labels.

*/
void TestToTensor(const Dataset& dataset, const std::vector<int>& size, tensor& images, tensor& labels, ColorType ctype = ColorType::BGR);

/** @brief Load a batch into images and labels EDDL tensors.

@param[in] dataset DLDataset object listing all the samples of a split.
@param[in] size Dimensions (width and height) at which all the images have to be resized.
@param[out] images Tensor which contains all the images.
@param[out] labels Tensor which contains all the labels.

*/
void LoadBatch(DLDataset& dataset, const std::vector<int>& size, tensor& images, tensor& labels);

} // namespace ecvl

#endif // ECVL_EDDL_H_