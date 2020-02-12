/*
* ECVL - European Computer Vision Library
* Version: 0.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#include "ecvl/core/support_opencv.h"

#include "ecvl/core/standard_errors.h"

namespace ecvl {
Image MatToImage(const cv::Mat& m)
{
    Image img;

    if (m.empty())
        return img;

    // https://stackoverflow.com/questions/7701921/free-cvmat-without-releasing-memory
    if (m.isContinuous()) {
        // Dims
        img.dims_ = std::vector<int>(m.dims);
        std::reverse_copy(m.size.p, m.size.p + m.dims, begin(img.dims_)); // OpenCV dims are {[, PLANES (DEPTH)], ROWS (HEIGHT), COLS(WIDTH)}

        // Type
        switch (m.depth()) {
        case CV_8U:  img.elemtype_ = DataType::uint8; break;
        case CV_8S:  img.elemtype_ = DataType::int8; break;
        case CV_16U: img.elemtype_ = DataType::uint16; break;
        case CV_16S: img.elemtype_ = DataType::int16; break;
        case CV_32S: img.elemtype_ = DataType::int32; break;
        case CV_32F: img.elemtype_ = DataType::float32; break;
        case CV_64F: img.elemtype_ = DataType::float64; break;
        default:
            ECVL_ERROR_UNSUPPORTED_OPENCV_DEPTH
        }
        img.elemsize_ = DataTypeSize(img.elemtype_);

        // Channels and colors
        if (m.dims < 2) {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }
        else if (m.dims == 2) {
            img.channels_ = "xy";
        }
        else if (m.dims == 3) {
            img.channels_ = "xyz";
        }
        else {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }

        if (m.type() == CV_8UC1 || m.type() == CV_16UC1 || m.type() == CV_32FC1 || m.type() == CV_64FC1) { // Guess this is a gray level image
            img.channels_ += "c";
            img.dims_.push_back(1); // Add another dim for color planes (but it is one dimensional)
            img.colortype_ = ColorType::GRAY;
        }
        else if (m.type() == CV_8UC3 || m.type() == CV_16UC3 || m.type() == CV_32FC3 || m.type() == CV_64FC3 ) { // Guess this is a BGR image
            img.channels_ += "c";
            img.dims_.push_back(3); // Add another dim for color planes
            img.colortype_ = ColorType::BGR;
        }
        else if (m.channels() == 1) {
            img.colortype_ = ColorType::none;
        }
        else {
            img.channels_ += "o";
            img.dims_.push_back(m.channels()); // Add another dim for color planes
            img.colortype_ = ColorType::none;
        }

        // Strides
        img.strides_.push_back(img.elemsize_);
        int dsize = img.dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            img.strides_.push_back(img.strides_[i] * img.dims_[i]);
        }

        // Data
        img.datasize_ = img.elemsize_;
        img.datasize_ = std::accumulate(begin(img.dims_), end(img.dims_), img.datasize_, std::multiplies<int>());
        img.mem_ = DefaultMemoryManager::GetInstance();
        img.data_ = img.mem_->Allocate(img.datasize_);
        // The following code copies the data twice. Should be improved!
        std::vector<cv::Mat> ch;
        cv::split(m, ch);
        std::vector<int> coords(img.dims_.size(), 0);
        int chsize = ch.size();
        for (int i = 0; i < chsize; ++i) {
            const cv::Mat& c = ch[i];
            coords.back() = i;
            //memcpy(img.data_ + (c.dataend - c.datastart) * i, c.data, c.dataend - c.datastart);
            memcpy(img.Ptr(coords), c.data, c.dataend - c.datastart);
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    return img;
}

cv::Mat ImageToMat(const Image& img)
{
    if (img.channels_ != "cxy" && img.channels_ != "xyc" &&
        img.channels_ != "zxy" && img.channels_ != "xyz" &&
        img.channels_ != "oxy" && img.channels_ != "xyo") {
        ECVL_ERROR_NOT_IMPLEMENTED
    }
    if (img.colortype_ != ColorType::BGR && img.colortype_ != ColorType::GRAY) {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    Image tmp;
    if (img.channels_.find('c') != std::string::npos) {
        RearrangeChannels(img, tmp, "cxy");
    }
    else if (img.channels_.find('z') != std::string::npos) {
        RearrangeChannels(img, tmp, "zxy");
    }
    else if (img.channels_.find('o') != std::string::npos) {
        RearrangeChannels(img, tmp, "oxy");
    }

    int type;
    switch (tmp.elemtype_) {
    case DataType::uint8:   type = CV_MAKETYPE(CV_8U, tmp.dims_[0]); break;
    case DataType::int8:    type = CV_MAKETYPE(CV_8S, tmp.dims_[0]); break;
    case DataType::uint16:  type = CV_MAKETYPE(CV_16U, tmp.dims_[0]); break;
    case DataType::int16:   type = CV_MAKETYPE(CV_16S, tmp.dims_[0]); break;
    case DataType::int32:   type = CV_MAKETYPE(CV_32S, tmp.dims_[0]); break;
    case DataType::float32: type = CV_MAKETYPE(CV_32F, tmp.dims_[0]); break;
    case DataType::float64: type = CV_MAKETYPE(CV_64F, tmp.dims_[0]); break;
    default:
        break;
    }

    int mdims[] = { tmp.dims_[2], tmp.dims_[1] }; // Swap dimensions to have rows, cols

    cv::Mat m(2, mdims, type);
    memcpy(m.data, tmp.data_, tmp.datasize_);

    return m;
}

Image MatVecToImage(const std::vector<cv::Mat>& v)
{
    Image img;

    if (v.empty())
        return img;

    if (v[0].isContinuous()) {
        // Dims
        img.dims_ = std::vector<int>(v[0].dims + 1);
        std::reverse_copy(v[0].size.p, v[0].size.p + v[0].dims, begin(img.dims_)); // OpenCV dims are {[, PLANES (DEPTH)], ROWS (HEIGHT), COLS(WIDTH)}
        img.dims_.back() = v.size();

        // Type
        switch (v[0].depth()) {
        case CV_8U:  img.elemtype_ = DataType::uint8; break;
        case CV_8S:  img.elemtype_ = DataType::int8; break;
        case CV_16U: img.elemtype_ = DataType::uint16; break;
        case CV_16S: img.elemtype_ = DataType::int16; break;
        case CV_32S: img.elemtype_ = DataType::int32; break;
        case CV_32F: img.elemtype_ = DataType::float32; break;
        case CV_64F: img.elemtype_ = DataType::float64; break;
        default:
            ECVL_ERROR_UNSUPPORTED_OPENCV_DEPTH
        }
        img.elemsize_ = DataTypeSize(img.elemtype_);

        // Channels and colors
        if (v[0].dims < 2) {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }
        else if (v[0].dims == 2) {
            img.channels_ = "xyz";
        }
        else if (v[0].dims == 3) {
            img.channels_ = "xyzw";
        }
        else {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }

        if (v[0].type() == CV_8UC1) { // Guess this is a gray level image
            img.channels_ += "c";
            img.dims_.push_back(1); // Add another dim for color planes (but it is one dimensional)
            img.colortype_ = ColorType::GRAY;
        }
        else if (v[0].type() == CV_8UC3) { // Guess this is a BGR image
            img.channels_ += "c";
            img.dims_.push_back(3); // Add another dim for color planes
            img.colortype_ = ColorType::BGR;
        }
        else if (v[0].channels() == 1) {
            img.colortype_ = ColorType::none;
        }
        else {
            img.channels_ += "o";
            img.dims_.push_back(v[0].channels()); // Add another dim for color planes
            img.colortype_ = ColorType::none;
        }

        // Strides
        img.strides_.push_back(img.elemsize_);
        int dsize = img.dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            img.strides_.push_back(img.strides_[i] * img.dims_[i]);
        }

        // Data
        img.datasize_ = img.elemsize_;
        img.datasize_ = std::accumulate(begin(img.dims_), end(img.dims_), img.datasize_, std::multiplies<int>());
        img.mem_ = DefaultMemoryManager::GetInstance();
        img.data_ = img.mem_->Allocate(img.datasize_);
        // The following code copies the data twice. Should be improved!

        std::vector<cv::Mat> channels;
        // For every channel
        for (int i = 0; i < img.dims_.back(); i++) {
            // For every slice
            for (size_t j = 0; j < v.size(); j++) {
                cv::split(v[j], channels);

                memcpy(img.data_ + i * img.strides_.back() + j * img.strides_[img.strides_.size() - 2], channels[i].data, img.strides_[img.strides_.size() - 2]);
            }
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    return img;
}
} // namespace ecvl 