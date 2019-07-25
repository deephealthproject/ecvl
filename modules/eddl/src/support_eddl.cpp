#include <ecvl/eddl.h>

#include "ecvl/core/imgproc.h"
#include "ecvl/core/imgcodecs.h"

#include <functional>
#include <iostream>
#include <cmath>

namespace ecvl
{
    Image TensorToImage(tensor& t)
    {
        if (t->data->ndim == 2)
        {
            Image img({ (int)sqrt(t->data->shape[1]), (int)sqrt(t->data->shape[1]), 1 }, DataType::float32, "xyc", ColorType::GRAY);
            memcpy(img.data_, t->input->ptr, img.datasize_);
            return img;
        }
        else
        {
            Image img({ t->input->shape }, DataType::float32, "xyc", ColorType::BGR);
            memcpy(img.data_, t->input->ptr, img.datasize_);
            return img;
        }
    }

    tensor ImageToTensor(const Image& img)
    {
        Image tmp;
        CopyImage(img, tmp, DataType::float32);

        if (tmp.channels_ != "xyc")
            RearrangeChannels(tmp, tmp, "xyc");

        int tot_dims = tmp.dims_[0] * tmp.dims_[1] * tmp.dims_[2];
        tensor t = eddl.T({ tot_dims });
        memcpy(t->input->ptr, tmp.data_, tmp.datasize_);
        return t;
    }

    tensor DatasetToTensor(vector<string> dataset, const std::vector<int>& dims, vector<AugFunctions>aug_f)
    {
        tensor t; Image img_tmp;

        //dims[0] = n_samples, dims[1] = n_channels, dims[2] = width, dims[3] = height
        int tot_dims = dims[1] * dims[2] * dims[3];
        tensor stack = eddl.T({ dims[0], tot_dims });
        int i = 0;
        cv::TickMeter tm;

        tm.reset();
        tm.start();
        for (auto& elem : dataset) {
            ImRead(elem, img_tmp);
            ResizeDim(img_tmp, img_tmp, { dims[2], dims[3] });

            //data augmentation
            for (auto f = aug_f.begin(); f != aug_f.end(); ++f)
                (*f)(img_tmp);

            t = ImageToTensor(img_tmp);
            memcpy(stack->input->ptr + t->input->size * i, t->input->ptr, t->input->size * sizeof(float));
            ++i;
            if (i % 100 == 0)
                std::cout << i << " images processed" << endl;
        }

        tm.stop();
        std::cout << "Elapsed " << tm.getTimeSec() << " s\n";
        return stack;
    }
}