#ifndef ECVL_CORE_H_
#define ECVL_CORE_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

#include "datatype.h"
#include "memorymanager.h"
#include "iterators.h"

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
public:
    DataType            elemtype_;
    uint8_t             elemsize_;
    std::vector<int>    dims_;      /**< Member description */
    std::vector<int>    strides_;
    std::string         channels_;
    ColorType           colortype_;
    uint8_t*            data_;
    size_t              datasize_;
    bool                contiguous_;

    MetaData* meta_;
    MemoryManager* mem_;


    template<typename T>
    auto Begin() { return Iterator<T>(*this); }
    template<typename T>
    auto End() { return Iterator<T>(*this, dims_); }

    template<typename T>
    auto Begin() const { return ConstIterator<T>(*this); }
    template<typename T>
    auto End() const { return ConstIterator<T>(*this, dims_); }

    template<typename T>
    auto ContiguousBegin() { return ContiguousIterator<T>(*this); }
    template<typename T>
    auto ContiguousEnd() { return ContiguousIterator<T>(*this, dims_); }

    template<typename T>
    auto ContiguousBegin() const { return ConstContiguousIterator<T>(*this); }
    template<typename T>
    auto ContiguousEnd() const { return ConstContiguousIterator<T>(*this, dims_); }

    /** @brief Default constructor

        The default constructor creates an empty image without any data.
    */
    Image() :
        elemtype_{ DataType::none },
        elemsize_{ DataTypeSize(elemtype_) },
        dims_{},
        strides_{},
        channels_{},
        colortype_{ ColorType::none },
        data_{ nullptr },
        datasize_{ 0 },
        contiguous_{ true },
        meta_{ nullptr },
        mem_{ nullptr }
    {
    }

    /** @brief Initializing constructor

        This constructor creates a proper image and allocates the data.
    */
    Image(std::initializer_list<int> dims, DataType elemtype, std::string channels, ColorType colortype) :
        elemtype_{ elemtype },
        elemsize_{ DataTypeSize(elemtype_) },
        dims_{dims},
        strides_{ elemsize_ },
        channels_{ move(channels) },
        colortype_{ colortype },
        data_{ nullptr },
        datasize_{ 0 },
        contiguous_{ true },
        meta_{ nullptr },
        mem_{ DefaultMemoryManager::GetInstance() }
    {
        // Compute strides of dimensions after 0
        int dsize = dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            strides_.push_back(strides_[i] * dims_[i]);
        }
        datasize_ = elemsize_;
        datasize_ = std::accumulate(begin(dims_), end(dims_), datasize_, std::multiplies<size_t>());
        data_ = mem_->Allocate(datasize_);
    }

    /** @brief Copy constructor: Deep Copy
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
        if (mem_)
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

#include "iterators_impl.h"

template <typename T>
class View : public Image {
public:
    View(Image& img) {
        elemtype_ = img.elemtype_;
        elemsize_ = img.elemsize_;
        dims_ = img.dims_;
        strides_ = img.strides_;
        channels_ = img.channels_;
        colortype_ = img.colortype_;
        data_ = img.data_;
        datasize_ = img.datasize_;
        contiguous_ = img.contiguous_;
        meta_ = img.meta_;
        mem_ = ShallowMemoryManager::GetInstance();
    }

    T& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<T*>(Ptr(coords));
    }

    auto Begin() { return Iterator<T>(*this); }
    auto End() { return Iterator<T>(*this, dims_); }
};

template <typename T>
class ConstView : public Image {
public:
    ConstView(const Image& img) {
        elemtype_ = img.elemtype_;
        elemsize_ = img.elemsize_;
        dims_ = img.dims_;
        strides_ = img.strides_;
        channels_ = img.channels_;
        colortype_ = img.colortype_;
        data_ = img.data_;
        datasize_ = img.datasize_;
        contiguous_ = img.contiguous_;
        meta_ = img.meta_;
        mem_ = ShallowMemoryManager::GetInstance();
    }

    const T& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<const T*>(Ptr(coords));
    }

    auto Begin() { return ConstIterator<T>(*this); }
    auto End() { return ConstIterator<T>(*this, dims_); }
};

template <typename T>
class ContiguousView : public Image {
public:
    ContiguousView(Image& img) {
        elemtype_ = img.elemtype_;
        elemsize_ = img.elemsize_;
        dims_ = img.dims_;
        strides_ = img.strides_;
        channels_ = img.channels_;
        colortype_ = img.colortype_;
        data_ = img.data_;
        datasize_ = img.datasize_;
        contiguous_ = img.contiguous_;
        meta_ = img.meta_;
        mem_ = ShallowMemoryManager::GetInstance();
    }

    T& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<T*>(Ptr(coords));
    }

    auto Begin() { return ContiguousIterator<T>(*this); }
    auto End() { return ContiguousIterator<T>(*this, dims_); }
};


} // namespace ecvl

#endif // !ECVL_CORE_H_


