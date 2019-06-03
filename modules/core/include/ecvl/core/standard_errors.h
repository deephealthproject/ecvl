#ifndef ECV_STANDARD_ERRORS_H_
#define ECV_STANDARD_ERRORS_H_

#define ECVL_ERROR_MSG "[Error]: "
#define ECVL_WARNING_MSG "[Warning]: "

#define ECVL_ERROR_NOT_IMPLEMENTED throw std::runtime_error(ECVL_ERROR_MSG "Not implemented");
#define ECVL_ERROR_NOT_REACHABLE_CODE throw std::runtime_error(ECVL_ERROR_MSG "How did you get here?");
#define ECVL_ERROR_WRONG_PARAMS(...) throw std::runtime_error(ECVL_ERROR_MSG "Wrong parameters - " __VA_ARGS__);
#define ECVL_ERROR_NOT_ALLOWED_ON_NON_OWING_IMAGE(...) throw std::runtime_error(ECVL_ERROR_MSG "Operation not allowed on non-owning Image" __VA_ARGS__);
#define ECVL_ERROR_UNSUPPORTED_OPENCV_DEPTH throw std::runtime_error(ECVL_ERROR_MSG "Unsupported OpenCV depth");
#define ECVL_ERROR_UNSUPPORTED_OPENCV_DIMS throw std::runtime_error(ECVL_ERROR_MSG "Unsupported OpenCV dimensions");

#endif // !ECV_STANDARD_ERRORS_H_
