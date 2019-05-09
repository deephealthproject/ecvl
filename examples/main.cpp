#include "../src/core.h"
#include "../src/imgcodecs.h"

using namespace ecvl;
using namespace filesystem;


/** @brief Brief description of the function/procedure.

@anchor value -> to set an invisible link that can be referred to inside the documentation using @ref value command

Complete description of the function/procedure

@note Here you can write special notes that will be displayed differently inside the final documentation (yellow bar on the left)

@param[in] m Description starting with capital letter
@param[out]
@param[in,out]

@return Description of the return value, None if void.
*/
Image MatToImage(cv::Mat& m)
{
    Image img;
    
    // https://stackoverflow.com/questions/7701921/free-cvmat-without-releasing-memory
    if (m.isContinuous()) {
        // Dims
        img.dims_ = std::vector<int>(m.dims);
        std::reverse_copy(m.size.p, m.size.p + m.dims, begin(img.dims_)); // OpenCV dims are {[, PLANES (DEPTH)], ROWS (HEIGHT), COLS(WIDTH)}
        if (m.channels() > 1) {
            img.dims_.push_back(m.channels());
        }

        // Type
        switch (m.depth()) {
        case CV_8U: img.elemtype_ = DataType::uint8; break;
        case CV_8S: img.elemtype_ = DataType::int8; break;
        case CV_16U: img.elemtype_ = DataType::uint16; break;
        case CV_16S: img.elemtype_ = DataType::int16; break;
        case CV_32S: img.elemtype_ = DataType::int32; break;
        case CV_32F: img.elemtype_ = DataType::float32; break;
        case CV_64F: img.elemtype_ = DataType::float64; break;
        default:
            throw std::runtime_error("Unsupported OpenCV Depth");
        }
        img.elemsize_ = DataTypeSize(img.elemtype_);

        // Strides
        img.strides_.push_back(img.elemsize_);
        int dsize = img.dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            img.strides_.push_back(img.strides_[i] * img.dims_[i]);
        }

        // Data
        img.datasize_ = std::accumulate(begin(img.dims_), end(img.dims_), img.elemsize_, std::multiplies<int>());
        img.data_ = new uint8_t[img.datasize_];
        // The following code copies the data twice. Should be improved!
        std::vector<cv::Mat> ch;
        cv::split(m, ch);
        std::vector<int> coords(img.dims_.size(), 0);
        int chsize = ch.size();
        for (int i = 0; i < chsize; ++i) {
            const cv::Mat& c = ch[i];
            coords.back() = i;
            //memcpy(img.data_ + (c.dataend - c.datastart) * i, c.data, c.dataend - c.datastart);
            memcpy(img.ptr(coords), c.data, c.dataend - c.datastart);
        }
    }
    else {
        throw std::runtime_error("Not implemented");
    }

    return img;
}

/*
cv::Mat ImageToMat(const Image& i) {

}
*/

#include <iostream>

int main(void)
{
    cv::Mat3b m(3, 2);
    m << cv::Vec3b(1, 1, 1), cv::Vec3b(2, 2, 2),
        cv::Vec3b(3, 3, 3), cv::Vec3b(4, 4, 4),
        cv::Vec3b(5, 5, 5), cv::Vec3b(6, 6, 6);
    Image img = MatToImage(m);

    std::cout << m;

    m = cv::imread("test.jpg");
    img = MatToImage(m);

    //Image j = ecvl::ImRead(path("test.jpg"));
    return 0;
}