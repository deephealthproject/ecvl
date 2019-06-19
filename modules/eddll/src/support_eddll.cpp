#include <ecvl/eddll.h>

#include "ecvl/core/imgproc.h"
#include "ecvl/core/imgcodecs.h"

namespace ecvl
{
    Image TensorToImage(tensor& t)
    {
        Image img({ t->input->shape }, DataType::float32, "xyc", ColorType::BGR);
        memcpy(img.data_, t->input->ptr, img.datasize_);
        return img;
    }

    tensor ImageToTensor(const Image& img)
    {
        Image tmp;
        CopyImage(img, tmp, DataType::float32);

        if (tmp.channels_ != "xyc")
            RearrangeChannels(tmp, tmp, "xyc");

        tensor t = eddl.T({ tmp.dims_ });
        memcpy(t->input->ptr, tmp.data_, tmp.datasize_);
        return t;
    }

    tensor DatasetToTensor(vector<string> dataset, const std::vector<int>& dims)
    {
        tensor t; Image img_tmp;

        //dims[0] = n_samples, dims[1] = n_channels, dims[2] = width, dims[3] = height
        tensor stack = eddl.T({ dims });
        int i = 0;
        for (auto& elem : dataset) {
            ImRead(elem, img_tmp);
            ResizeDim(img_tmp, img_tmp, { dims[2], dims[3] });
            t = ImageToTensor(img_tmp);
            memcpy(stack->input->ptr + t->input->size * i, t->input->ptr, t->input->size * sizeof(float));
            ++i;
        }
        return stack;
    }
}