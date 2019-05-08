#ifndef ECVL_CORE_H_
#define ECVL_CORE_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

#include "datatype.h"

namespace ecvl {

struct MetaData {
    virtual bool Query(const std::string& name, std::string& value) const = 0;
    virtual ~MetaData() {};
};

struct BaseImage {
    std::vector<int> dims_;      /**< Member description */
    std::vector<int> strides_;
    DataType elemtype_;
    uint8_t elemsize_;
    uint8_t* data_ = nullptr;
    size_t datasize_ = 0;
    bool contiguous_ = true;

    MetaData* meta_ = nullptr;

    BaseImage() {}
    BaseImage(std::initializer_list<int> dims, DataType elemtype) : dims_{ dims }, strides_{ elemsize_ }, elemtype_{ elemtype } {
        elemsize_ = DataTypeSize(elemtype);
        int dsize = dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            strides_.push_back(strides_[i] * dims_[i]);
        }
        datasize_ = elemsize_;
        datasize_ = std::accumulate(begin(dims_), end(dims_), datasize_, std::multiplies<size_t>());
    }
};

struct Image : public BaseImage {
    Image() {}

    Image(std::initializer_list<int> dims, DataType elemtype) : 
        BaseImage(dims, elemtype)
    {
        data_ = new uint8_t[datasize_];
    }

    Image(int width, int height, DataType elemtype) :
        Image({ width, height }, elemtype) {}

    Image(const Image& img) : BaseImage(img) {
        data_ = new uint8_t[datasize_];
        memcpy(data_, img.data_, datasize_);
    }

    Image(Image&& img) : BaseImage(std::move(img)) {
        img.data_ = nullptr;
    }

    friend void swap(Image& lhs, Image& rhs) {
        using std::swap;
        swap(lhs.dims_, rhs.dims_);
        swap(lhs.strides_, rhs.strides_);
        swap(lhs.elemtype_, rhs.elemtype_);
        swap(lhs.elemsize_, rhs.elemsize_);
        swap(lhs.data_, rhs.data_);
        swap(lhs.datasize_, rhs.datasize_);
        swap(lhs.contiguous_, rhs.contiguous_);
    }

    Image& operator=(Image rhs) {
        swap(*this, rhs);
        return *this;
    }

    ~Image() {
        delete[] data_;
    }

    uint8_t* ptr(const std::vector<int>& coords) {
        assert(coords.size() == strides_.size());
        return std::inner_product(begin(coords), end(coords), begin(strides_), data_);
    }
};

struct ImageView : public BaseImage {
    ImageView(Image& img) : BaseImage(img) 
    {
        data_ = img.data_;
    }
};

/*
template<typename T>
struct Image_ : Image {
    //std::vector<int> dims_;
    //std::vector<int> strides_;
    //T* data_ = nullptr;
    //bool owned_ = true;
    //
    //MetaData* meta_ = nullptr;

    //Image() {}
    //Image(int width, int height) : 
    //    dims_{ width, height }, 
    //    strides_{ 1, width }, 
    //    data_{ new T[width*height] }
    //    {}
    //// Come distinguiamo depth da spectrum (canali)? 
    //Image(int width, int height, int depth) :
    //    dims_{ width, height, depth },
    //    strides_{ 1, width, width*height },
    //    data_{ new T[width*height*depth] }
    //{}
    //Image(std::initializer_list<int> dims) : dims_(dims), strides_{1} {
    //    for (int i = 0; i < dims_.size() - 1; ++i) {
    //        strides_.push_back(strides_[i] * dims_[i]);
    //    }
    //}
};
*/
} // namespace ecvl

#endif // !ECVL_CORE_H_


