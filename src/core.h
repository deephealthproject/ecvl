#ifndef ECVL_CORE_H_
#define ECVL_CORE_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

#include "datatype.h"
#include "memorymanager.h"

namespace ecvl {

class MetaData {
public:
    virtual bool Query(const std::string& name, std::string& value) const = 0;
    virtual ~MetaData() {}
};

enum class ColorType {
    none,
    GRAY,
    RGB,
    BGR,
    HSV,
    YCbCr,
};

class Image {

    void InitInformation() {
        strides_.push_back(elemsize_);
        int dsize = dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            strides_.push_back(strides_[i] * dims_[i]);
        }
        datasize_ = elemsize_;
        datasize_ = std::accumulate(begin(dims_), end(dims_), datasize_, std::multiplies<size_t>());
    }

public:
    DataType            elemtype_    = DataType::none;
    uint8_t             elemsize_    = 0;
    std::vector<int>    dims_;      /**< Member description */
    std::vector<int>    strides_;
    std::string         channels_;
    ColorType           colortype_   = ColorType::none;
    uint8_t*            data_        = nullptr;
    size_t              datasize_    = 0;
    bool                contiguous_  = true;

    MetaData* meta_ = nullptr;
    MemoryManager* mem_ = DefaultMemoryManager::GetInstance();

    Image() {}
    Image(std::initializer_list<int> dims, DataType elemtype, std::string channels, ColorType colortype) : 
        elemtype_{ elemtype },
        elemsize_{ DataTypeSize(elemtype_) },
        dims_{dims},
        channels_{ move(channels) }, 
        colortype_{ colortype }
    {
        InitInformation();
        data_ = mem_->Allocate(datasize_);
    }

//    Image(int width, int height, DataType elemtype) :
//        Image({ width, height }, elemtype) {}

    /** @brief Copy constructor: Deep Copy

        Copy constructor: Deep Copy
    */
    Image(const Image& img) :
        elemtype_{ img.elemtype_ },
        elemsize_{ img.elemsize_ },
        dims_{ img.dims_ },
        strides_{ img.strides_ },
        channels_{ img.channels_ }, 
        colortype_{ img.colortype_ }, 
        data_{},
        datasize_{ img.datasize_ },
        contiguous_{ img.contiguous_ },
        meta_{ img.meta_ },
        mem_{ img.mem_ }
    {
        data_ = mem_->AllocateAndCopy(datasize_, img.data_);
    }

    /** @brief Move constructor

        Move constructor
    */
    Image(Image&& img) :
        elemtype_{ img.elemtype_ },
        elemsize_{ img.elemsize_ },
        dims_{ move(img.dims_) },
        strides_{ move(img.strides_) },
        channels_{ move(img.channels_) },
        colortype_{ img.colortype_ },
        data_{ img.data_ },
        datasize_{ img.datasize_ },
        contiguous_{ img.contiguous_ },
        meta_{ img.meta_ },
        mem_{ img.mem_ }
    {
        img.data_ = nullptr;
    }

    friend void swap(Image& lhs, Image& rhs) {
        using std::swap;
        swap(lhs.elemtype_, rhs.elemtype_);
        swap(lhs.elemsize_, rhs.elemsize_);
        swap(lhs.dims_, rhs.dims_);
        swap(lhs.strides_, rhs.strides_);
        swap(lhs.channels_, rhs.channels_);
        swap(lhs.colortype_, rhs.colortype_);
        swap(lhs.data_, rhs.data_);
        swap(lhs.datasize_, rhs.datasize_);
        swap(lhs.contiguous_, rhs.contiguous_);
        swap(lhs.meta_, rhs.meta_);
        swap(lhs.mem_, rhs.mem_);
    }

    Image& operator=(Image rhs) {
        swap(*this, rhs);
        return *this;
    }

    ~Image() {
        mem_->Deallocate(data_);
    }

    bool IsEmpty() const { return data_ == nullptr; }

    uint8_t* Ptr(const std::vector<int>& coords) {
        assert(coords.size() == strides_.size());
        return std::inner_product(begin(coords), end(coords), begin(strides_), data_);
    }
    const uint8_t* Ptr(const std::vector<int>& coords) const {
        assert(coords.size() == strides_.size());
        return std::inner_product(begin(coords), end(coords), begin(strides_), data_);
    }
};

template <typename T>
class Img : public Image {
public:
    Img(Image& img) : Image(img) {}

    T& operator()(int x, int y) {
        return *reinterpret_cast<T*>(data_ + x*strides_[0] + y * strides_[1]);
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


