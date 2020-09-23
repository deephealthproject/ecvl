/*
* ECVL - European Computer Vision Library
* Version: 0.2.3
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_METADATA_H_
#define ECVL_METADATA_H_

#include <unordered_map>
#include <memory>

namespace ecvl
{
class MetaSingleBase
{
public:
    virtual ~MetaSingleBase() {}
    template<class T> const T& Query() const;
};

template <typename T>
class MetaSingle : public MetaSingleBase
{
    T value_;
public:
    MetaSingle(const T& value) : value_(value) {}
    const T& Query() const { return value_; }
};

template<class T> const T& MetaSingleBase::Query() const
{
    return static_cast<const MetaSingle<T>&>(*this).Query();
}

class MetaData
{
public:
    std::unordered_map<std::string, std::shared_ptr<MetaSingleBase>> map_;
};
} // namespace ecvl

#endif // !ECVL_METADATA_H_
