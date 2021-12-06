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

#include "ecvl/core/support_opencv.h"

#include "ecvl/core/imgproc.h"
#include "ecvl/core/standard_errors.h"

namespace ecvl
{
Image MatToImage(const cv::Mat& m, ColorType ctype, const std::string& dst_channels)
{
    Image img;

    if (m.empty())
        return img;

    // https://stackoverflow.com/questions/7701921/free-cvmat-without-releasing-memory
    if (m.isContinuous()) {
        // Dims
        std::vector<int> dims(m.dims);
        std::reverse_copy(m.size.p, m.size.p + m.dims, begin(dims)); // OpenCV dims are {[, PLANES (DEPTH)], ROWS (HEIGHT), COLS(WIDTH)}

        // Type
        DataType elemtype;
        switch (m.depth()) {
        case CV_8U:  elemtype = DataType::uint8; break;
        case CV_8S:  elemtype = DataType::int8; break;
        case CV_16U: elemtype = DataType::uint16; break;
        case CV_16S: elemtype = DataType::int16; break;
        case CV_32S: elemtype = DataType::int32; break;
        case CV_32F: elemtype = DataType::float32; break;
        case CV_64F: elemtype = DataType::float64; break;
        default:
            ECVL_ERROR_UNSUPPORTED_OPENCV_DEPTH
        }

        // Channels and colors
        std::string channels;
        if (m.dims < 2) {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }
        else if (m.dims == 2) {
            channels = "xy";
        }
        else if (m.dims == 3) {
            channels = "xyz";
        }
        else {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }

        ColorType colortype;
        if (m.type() == CV_8UC1 || m.type() == CV_16UC1 || m.type() == CV_32FC1 || m.type() == CV_64FC1
            || m.type() == CV_8SC1 || m.type() == CV_16SC1 || m.type() == CV_32SC1) { // Guess this is a gray level image
            channels += "c";
            dims.push_back(1); // Add another dim for color planes (but it is one dimensional)
            colortype = ColorType::GRAY;
        }
        else if (m.type() == CV_8UC3 || m.type() == CV_16UC3 || m.type() == CV_32FC3 || m.type() == CV_64FC3
            || m.type() == CV_8SC3 || m.type() == CV_16SC3 || m.type() == CV_32SC3) { // Guess this is a BGR image
            channels += "c";
            dims.push_back(3); // Add another dim for color planes
            colortype = ColorType::BGR;
        }
        else if (m.channels() == 1) {
            colortype = ColorType::none;
        }
        else {
            channels += "o";
            dims.push_back(m.channels()); // Add another dim for color planes
            colortype = ColorType::none;
        }

        img.Create(dims, elemtype, channels, colortype);

        // The following code copies the data twice. Should be improved!
        std::vector<cv::Mat> ch;
        cv::split(m, ch);
        std::vector<int> coords(img.dims_.size(), 0);
        int chsize = vsize(ch);
        for (int i = 0; i < chsize; ++i) {
            const cv::Mat& c = ch[i];
            coords.back() = i;
            //memcpy(img.data_ + (c.dataend - c.datastart) * i, c.data, c.dataend - c.datastart);
            img.hal_->MemCopy(img.Ptr(coords), c.data, c.dataend - c.datastart);
        }

        // Switch to specific colortype if given
        if (ctype != ColorType::none && ctype != img.colortype_) {
            ChangeColorSpace(img, img, ctype);
        }

        if (dst_channels != "") {
            RearrangeChannels(img, img, dst_channels);
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    return img;
}

cv::Mat ImageToMat(const Image& img)
{
    if (!(img.Width() && img.Height() && img.Channels() && vsize(img.dims_) == 3 && img.elemtype_ != DataType::int64)) {
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

    if (img.colortype_ == ColorType::RGB) {
        ChangeColorSpace(tmp, tmp, ColorType::BGR);
    }
    else if (img.colortype_ != ColorType::BGR && img.colortype_ != ColorType::GRAY && img.colortype_ != ColorType::none) {
        ECVL_ERROR_NOT_IMPLEMENTED
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
    //memcpy(m.data, tmp.data_, tmp.datasize_);
    img.hal_->MemCopy(m.data, tmp.data_, tmp.datasize_);

    return m;
}

Image MatVecToImage(const std::vector<cv::Mat>& v)
{
    Image img;

    if (v.empty())
        return img;

    if (v[0].isContinuous()) {
        // Dims
        std::vector<int> dims(v[0].dims + 1);
        std::reverse_copy(v[0].size.p, v[0].size.p + v[0].dims, begin(dims)); // OpenCV dims are {[, PLANES (DEPTH)], ROWS (HEIGHT), COLS(WIDTH)}
        dims.back() = vsize(v);

        // Type
        DataType elemtype;
        switch (v[0].depth()) {
        case CV_8U:  elemtype = DataType::uint8; break;
        case CV_8S:  elemtype = DataType::int8; break;
        case CV_16U: elemtype = DataType::uint16; break;
        case CV_16S: elemtype = DataType::int16; break;
        case CV_32S: elemtype = DataType::int32; break;
        case CV_32F: elemtype = DataType::float32; break;
        case CV_64F: elemtype = DataType::float64; break;
        default:
            ECVL_ERROR_UNSUPPORTED_OPENCV_DEPTH
        }

        // Channels and colors
        std::string channels;
        if (v[0].dims < 2) {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }
        else if (v[0].dims == 2) {
            channels = "xyz";
        }
        else if (v[0].dims == 3) {
            channels = "xyzw";
        }
        else {
            ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS
        }

        ColorType colortype;
        if (v[0].type() == CV_8UC1) { // Guess this is a gray level image
            channels += "c";
            dims.push_back(1); // Add another dim for color planes (but it is one dimensional)
            colortype = ColorType::GRAY;
        }
        else if (v[0].type() == CV_8UC3) { // Guess this is a BGR image
            channels += "c";
            dims.push_back(3); // Add another dim for color planes
            colortype = ColorType::BGR;
        }
        else if (v[0].channels() == 1) {
            colortype = ColorType::none;
        }
        else {
            channels += "o";
            dims.push_back(v[0].channels()); // Add another dim for color planes
            colortype = ColorType::none;
        }

        img.Create(dims, elemtype, channels, colortype);

        // The following code copies the data twice. Should be improved!
        std::vector<cv::Mat> vchannels;
        // For every channel
        for (int i = 0; i < img.dims_.back(); i++) {
            // For every slice
            for (size_t j = 0; j < v.size(); j++) {
                cv::split(v[j], vchannels);

                /*memcpy(img.data_ + i * img.strides_.back() + j * img.strides_[img.strides_.size() - 2],
                    vchannels[i].data, img.strides_[img.strides_.size() - 2]);*/
                img.hal_->MemCopy(img.data_ + i * img.strides_.back() + j * img.strides_[img.strides_.size() - 2],
                    vchannels[i].data, img.strides_[img.strides_.size() - 2]);
            }
        }
    }
    else {
        ECVL_ERROR_NOT_IMPLEMENTED
    }

    return img;
}
} // namespace ecvl 