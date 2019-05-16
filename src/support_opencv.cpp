#include "support_opencv.h"

ecvl::Image MatToImage(const cv::Mat& m)
{
    using namespace ecvl;

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
            throw std::runtime_error("Unsupported OpenCV Depth");
        }
        img.elemsize_ = DataTypeSize(img.elemtype_);

        // Channels and colors
        if (m.dims < 2) {
            throw std::runtime_error("Unsupported OpenCV dims");
        }
        else if (m.dims == 2) {
            img.channels_ = "xy";
        }
        else if (m.dims == 3) {
            img.channels_ = "xyz";
        }
        else {
            throw std::runtime_error("Unsupported OpenCV dims");
        }

        if (m.type() == CV_8UC1) { // Guess this is a gray level image
            img.channels_ += "c";
            img.dims_.push_back(1); // Add another dim for color planes (but it is one dimensional)
            img.colortype_ = ColorType::GRAY;
        }
        else if (m.type() == CV_8UC3) { // Guess this is a BGR image
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
        throw std::runtime_error("Not implemented");
    }

    return img;
}

cv::Mat ImageToMat(const ecvl::Image& img)
{
    using namespace ecvl;

    if (!img.contiguous_)
        throw std::runtime_error("Not implemented");

    cv::Mat m;
    if (img.channels_ == "xyc") {
        int type;
        switch (img.elemtype_)
        {
        case DataType::uint8:   type = CV_MAKETYPE(CV_8U, 1); break;
        case DataType::int8:    type = CV_MAKETYPE(CV_8S, 1); break;
        case DataType::uint16:  type = CV_MAKETYPE(CV_16U, 1); break;
        case DataType::int16:   type = CV_MAKETYPE(CV_16S, 1); break;
        case DataType::int32:   type = CV_MAKETYPE(CV_32S, 1); break;
        case DataType::float32: type = CV_MAKETYPE(CV_32F, 1); break;
        case DataType::float64: type = CV_MAKETYPE(CV_64F, 1); break;
        default:
            break;
        }

        int tmp[] = { img.dims_[1], img.dims_[0] }; // Swap dimensions to have rows, cols

        std::vector<cv::Mat> channels;
        for (int c = 0; c < img.dims_[2]; ++c) {
            channels.emplace_back(2, tmp, type, (void*)(img.Ptr({ 0,0,c })));
        }
        cv::merge(channels, m);
    }
    else {
        throw std::runtime_error("Not implemented");
    }

    return m;
}
