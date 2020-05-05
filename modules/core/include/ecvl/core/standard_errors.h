/*
* ECVL - European Computer Vision Library
* Version: 0.2.1
* copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
* Authors:
*    Costantino Grana (costantino.grana@unimore.it)
*    Federico Bolelli (federico.bolelli@unimore.it)
*    Michele Cancilla (michele.cancilla@unimore.it)
*    Laura Canalini (laura.canalini@unimore.it)
*    Stefano Allegretti (stefano.allegretti@unimore.it)
* All rights reserved.
*/

#ifndef ECVL_STANDARD_ERRORS_H_
#define ECVL_STANDARD_ERRORS_H_

#include <stdexcept>

#define ECVL_ERROR_MSG "[Error]: "
#define ECVL_WARNING_MSG "[Warning]: "

#define ECVL_ERROR_NOT_IMPLEMENTED throw std::runtime_error(ECVL_ERROR_MSG "Not implemented");
#define ECVL_ERROR_NOT_IMPLEMENTED_WHAT(what) throw std::runtime_error(ECVL_ERROR_MSG what " not implemented");
#define ECVL_ERROR_NOT_REACHABLE_CODE throw std::runtime_error(ECVL_ERROR_MSG "How did you get here?");
#define ECVL_ERROR_WRONG_PARAMS(msg) throw std::runtime_error(ECVL_ERROR_MSG "Wrong parameters - " msg);
#define ECVL_ERROR_NOT_ALLOWED_ON_NON_OWING_IMAGE(...) throw std::runtime_error(ECVL_ERROR_MSG "Operation not allowed on non-owning Image" __VA_ARGS__);
#define ECVL_ERROR_UNSUPPORTED_OPENCV_DEPTH throw std::runtime_error(ECVL_ERROR_MSG "Unsupported OpenCV depth");
#define ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS throw std::runtime_error(ECVL_ERROR_MSG "Unsupported OpenCV dimensions");
#define ECVL_ERROR_EMPTY_IMAGE throw std::runtime_error(ECVL_ERROR_MSG "Empty image provided");
#define ECVL_ERROR_NOT_ALLOWED_ON_UNSIGNED_IMG throw std::runtime_error(ECVL_ERROR_MSG "Operation not allowed on unsigned Image");
#define ECVL_ERROR_DIVISION_BY_ZERO throw std::runtime_error(ECVL_ERROR_MSG "Division by zero is not allowed.");
#define ECVL_ERROR_FILE_DOES_NOT_EXIST throw std::runtime_error(ECVL_ERROR_MSG "File does not exist");
#define ECVL_ERROR_SPLIT_DOES_NOT_EXIST throw std::runtime_error(ECVL_ERROR_MSG "This split does not exist in the dataset file");
#define ECVL_ERROR_CANNOT_LOAD_FROM_URL throw std::runtime_error(ECVL_ERROR_MSG "Cannot load from URL");
#define ECVL_ERROR_CANNOT_LOAD_IMAGE throw std::runtime_error(ECVL_ERROR_MSG "Cannot load image");
#define ECVL_ERROR_INCOMPATIBLE_DIMENSIONS throw std::runtime_error(ECVL_ERROR_MSG "Incompatible dimensions");
#define ECVL_ERROR_DIFFERENT_DEVICES throw std::runtime_error(ECVL_ERROR_MSG "src and dst are in different devices");
#define ECVL_ERROR_DEVICE_UNAVAILABLE(device) throw std::runtime_error(ECVL_ERROR_MSG #device " device unavailable");
#define ECVL_ERROR_MOVING_IMAGE(str, device) throw std::runtime_error(ECVL_ERROR_MSG "Cannot move " #str " " #device);


#endif // ECVL_STANDARD_ERRORS_H_
