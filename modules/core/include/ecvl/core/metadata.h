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

#include <typeindex>
#include "ecvl/core/any.h"

namespace ecvl
{
class MetaData
{
    std::any value_;
    std::string value_str_ = "";

    static inline std::unordered_map<std::type_index, std::function<void(const std::any& value, std::string& s)>> anytype_to_string{
    {std::type_index(typeid(std::string)), [](const std::any& x, std::string& s) {s = std::any_cast<std::string>(x); }},
    {std::type_index(typeid(int)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<int>(x)); }},
    {std::type_index(typeid(float)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<float>(x)); }},
    {std::type_index(typeid(double)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<double>(x)); }},
    {std::type_index(typeid(long)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<long>(x)); }},
    {std::type_index(typeid(long long)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<long long>(x)); }},
    {std::type_index(typeid(short)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<short>(x)); }},
    {std::type_index(typeid(unsigned)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<unsigned>(x)); }},
    {std::type_index(typeid(unsigned int)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<unsigned int>(x)); }},
    {std::type_index(typeid(unsigned short)), [](const std::any& x, std::string& s) {s = std::to_string(std::any_cast<unsigned short>(x)); }},
    };

public:
    // int is a workaround for some version of gcc bug: https://stackoverflow.com/questions/64744074/why-an-object-is-not-constructible?rq=1
    MetaData(const std::any& value, int) : value_(value) {}
    std::any Get() const { return value_; }
    std::string GetStr()
    {
        if (const auto it = anytype_to_string.find(std::type_index(value_.type())); it != anytype_to_string.cend()) {
            it->second(value_, value_str_);
        }

        return value_str_;
    }
};
} // namespace ecvl

#endif // !ECVL_METADATA_H_
