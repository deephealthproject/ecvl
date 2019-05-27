#ifndef ECVL_IMAGE_H_
#define ECVL_IMAGE_H_

#include <algorithm>
#include <numeric>
#include <stdexcept>
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

/** @anchor ColorType

   Enum class representing the ECVL supported color spaces.

*/
enum class ColorType {
    none,
    GRAY,
    RGB,
    BGR,
    HSV,
    YCbCr,
};

/** @brief Image class

*/
class Image {
public:
    DataType            elemtype_;  /**< Type of Image pixels, must be one of the 
                                         values available in @ref DataType.        */
    uint8_t             elemsize_;  /**< Size (in bytes) of Image pixels.          */
    std::vector<int>    dims_;      /**< Vector of Image dimensions. Each dimension 
                                         is given in pixels/voxels. */
    std::vector<int>    strides_;   /**< Vector of Image strides. Strides represent
                                         the number of bytes the pointer on data 
                                         has to move to reach the next pixel/voxel 
                                         on the correspondent size. */
    std::string         channels_;  /**< String which describes how Image planes 
                                         are organized. A single character provides
                                         the information related to the corresponding
                                         channel. The possible values are:
                                            - 'x': horizontal spatial dimension
                                            - 'y': vertical spatial dimension
                                            - 'z': depth spatial dimension
                                            - 'c': color dimension
                                            - 't': temporal dimension
                                            - 'o': any other dimension
                                         For example, "xyc" describes a 2-dimensional
                                         Image structured in color planes. This could
                                         be for example a ColorType::GRAY Image with 
                                         dims_[2] = 1 or a ColorType::RGB Image with
                                         dims_[2] = 3 an so on. The ColorType constrains
                                         the value of the dimension corresponding to 
                                         the color channel. 
                                         Another example is "cxy" with dims_[0] = 3 and
                                         ColorType::BGR. In this case the color dimension
                                         is the one which changes faster as it is done 
                                         in other libraries such as OpenCV. */
    ColorType           colortype_; /**< Image ColorType. If this is different from ColorType::none
                                         the channels_ string must contain a 'c' and the
                                         corresponding dimension must have the appropriate
                                         value. See @ref ColorType for the possible values. */
    uint8_t*            data_;
    size_t              datasize_;
    bool                contiguous_;

    MetaData* meta_;
    MemoryManager* mem_;

    template<typename T>
    Iterator<T> Begin() { return Iterator<T>(*this); }
    template<typename T>
    Iterator<T> End() { return Iterator<T>(*this, dims_); }

    template<typename T>
    ConstIterator<T> Begin() const { return ConstIterator<T>(*this); }
    template<typename T>
    ConstIterator<T> End() const { return ConstIterator<T>(*this, dims_); }

    template<typename T>
    ContiguousIterator<T> ContiguousBegin() { return ContiguousIterator<T>(*this); }
    template<typename T>
    ContiguousIterator<T> ContiguousEnd() { return ContiguousIterator<T>(*this, dims_); }

    template<typename T>
    ConstContiguousIterator<T> ContiguousBegin() const { return ConstContiguousIterator<T>(*this); }
    template<typename T>
    ConstContiguousIterator<T> ContiguousEnd() const { return ConstContiguousIterator<T>(*this, dims_); }

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
    Image(const std::vector<int>& dims, DataType elemtype, std::string channels, ColorType colortype) :
        elemtype_{ elemtype },
        elemsize_{ DataTypeSize(elemtype_) },
        dims_{dims},
        strides_{},
        channels_{ move(channels) },
        colortype_{ colortype },
        data_{ nullptr },
        datasize_{ 0 },
        contiguous_{ true },
        meta_{ nullptr },
        mem_{ DefaultMemoryManager::GetInstance() }
    {
        // Compute strides
        strides_ = { elemsize_ };
        int dsize = dims_.size();
        for (int i = 0; i < dsize - 1; ++i) {
            strides_.push_back(strides_[i] * dims_[i]);
        }
        // Compute datasize
        datasize_ = elemsize_;
        datasize_ = std::accumulate(begin(dims_), end(dims_), datasize_, std::multiplies<size_t>());
        data_ = mem_->Allocate(datasize_);
    }

    /** @brief Copy constructor: Deep Copy
    */
    Image(const Image& img) :
        elemtype_{ img.elemtype_ },//
        elemsize_{ img.elemsize_ },//
        dims_{ img.dims_ },//
        strides_{ img.strides_ },//
        channels_{ img.channels_ },// 
        colortype_{ img.colortype_ },//
        data_{},//
        datasize_{ img.datasize_ },//
        contiguous_{ img.contiguous_ },//
        meta_{ img.meta_ },//
        mem_{ img.mem_ }
    {
        if (mem_ == ShallowMemoryManager::GetInstance()) {
            // When copying from non owning memory we become owners of the original data.
            mem_ = DefaultMemoryManager::GetInstance();
        }
        if (mem_ == DefaultMemoryManager::GetInstance()) {
            if (contiguous_) {
                data_ = mem_->AllocateAndCopy(datasize_, img.data_);
            }
            else {
                // When copying a non contiguous image, we make it contiguous
                contiguous_ = true;
                // Compute strides
                strides_ = { elemsize_ };
                int dsize = dims_.size();
                for (int i = 0; i < dsize - 1; ++i) {
                    strides_.push_back(strides_[i] * dims_[i]);
                }
                // Compute datasize
                datasize_ = elemsize_;
                datasize_ = std::accumulate(begin(dims_), end(dims_), datasize_, std::multiplies<size_t>());
                data_ = mem_->Allocate(datasize_);
                // Copy with iterators
                // TODO: optimize so that we can memcpy one block at a time on the first dimension
                // This will require Iterators to increment more than one
                auto p = data_;
                auto i = Begin<uint8_t>(), e = End<uint8_t>();
                for (; i != e; ++i) {
                    memcpy(p++, i.ptr_, elemsize_);
                }
            }
        }
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

#include "iterators_impl.inc"
template <DataType DT>
class View : public Image {
public:
    using basetype = typename TypeInfo<DT>::basetype;

    View(Image& img) 
    {
        if (DT != img.elemtype_)
            throw std::runtime_error("View type is different from Image type");
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

    View(Image& img, const std::vector<int>& start, const std::vector<int>& size) : View(img)
    {
        dims_.clear();
        int ssize = size.size();
        for (int i = 0; i < ssize; ++i) {
            if (start[i] < 0 || start[i] >= img.dims_[i])
                throw std::runtime_error("Start of crop outside image limits");
            dims_.push_back(img.dims_[i] - start[i]);
            if (size[i] > dims_[i]) {
                throw std::runtime_error("Crop outside image limits");
            }
            if (size[i] >= 0) {
                dims_[i] = size[i];
            }
        }

        // Check if image has a color dimension
        auto cpos = channels_.find('c');
        if (cpos != std::string::npos) {
            // If we are cropping the color channel, we fix the color information
            if (dims_[cpos] != img.dims_[cpos]) {
                if (dims_[cpos] == 1) {
                    colortype_ = ColorType::GRAY;
                }
                else {
                    channels_[cpos] = 'o';
                    colortype_ = ColorType::none;
                }
            }
        }

        data_ = img.Ptr(start);
        datasize_ = 0; // This is set to zero, because when the View is not contiguous, it's useless to relay on this information
        contiguous_ = false;
    }

    basetype& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<basetype*>(Ptr(coords));
    }

    Iterator<basetype> Begin() { return Iterator<basetype>(*this); }
    Iterator<basetype> End() { return Iterator<basetype>(*this, dims_); }
};

template <DataType DT>
class ConstView : public Image {
public:
    using basetype = typename TypeInfo<DT>::basetype;

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

    const basetype& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<const basetype*>(Ptr(coords));
    }

    ConstIterator<basetype> Begin() { return ConstIterator<basetype>(*this); }
    ConstIterator<basetype> End() { return ConstIterator<basetype>(*this, dims_); }
};

template <DataType DT>
class ContiguousView : public Image {
public:
    using basetype = typename TypeInfo<DT>::basetype;

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

    basetype& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<basetype*>(Ptr(coords));
    }

    ContiguousIterator<basetype> Begin() { return ContiguousIterator<basetype>(*this); }
    ContiguousIterator<basetype> End() { return ContiguousIterator<basetype>(*this, dims_); }
};

template <DataType DT>
class ConstContiguousView : public Image {
public:
    using basetype = typename TypeInfo<DT>::basetype;

    ConstContiguousView(Image& img) {
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

    const basetype& operator()(const std::vector<int>& coords) {
        return *reinterpret_cast<const basetype*>(Ptr(coords));
    }

    ConstContiguousIterator<basetype> Begin() { return ConstContiguousIterator<basetype>(*this); }
    ConstContiguousIterator<basetype> End() { return ConstContiguousIterator<basetype>(*this, dims_); }
};

/** @brief Changes the order of the Image dimensions.

The RearrangeChannels procedure changes the order of the input Image dimensions saving 
the result into the output Image. The new order of dimensions can be specified as a 
string through the "channels" parameter. Input and output Images can be the same. The
number of channels of the input Image must be the same of required channels.

@param[in] src Input Image on which to rearrange dimensions.
@param[out] dst The output rearranged Image. Can be the src Image.
@param[in] channels Desired order of Image channels.

*/
void RearrangeChannels(const Image& src, Image& dst, const std::string& channels);

} // namespace ecvl

#endif // !ECVL_IMAGE_H_


