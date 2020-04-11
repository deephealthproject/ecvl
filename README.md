
![ECVL](doc/logo/DEEPHEALTH_doxygen_logo_reduced.png)
# ECVL - European Computer Vision Library 
![release](https://img.shields.io/github/v/release/deephealthproject/ecvl)
[![docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat)](https://deephealthproject.github.io/ecvl/)
![cobertura](https://img.shields.io/jenkins/coverage/cobertura?jobUrl=https%3A%2F%2Fjenkins-master-deephealth-unix01.ing.unimore.it%2Fjob%2FDeepHealth%2Fjob%2Fecvl%2Fjob%2Fmaster%2F&label=cobertura)
[![codecov](https://codecov.io/gh/deephealthproject/ecvl/branch/master/graph/badge.svg)](https://codecov.io/gh/deephealthproject/ecvl)
[![license](https://img.shields.io/apm/l/vim-mode)](https://github.com/deephealthproject/ecvl/blob/master/LICENSE)<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![contributors](https://img.shields.io/badge/all_contributors-13-orange.svg?style=flat)](#contributors)<!-- ALL-CONTRIBUTORS-BADGE:END -->

| System  |  Compiler  | OpenCV | Status | 
|:-------:|:----------:|:------:|:------:|
| Windows (CPU) | VS 15.9.11 | 3.4.6  |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ecvl/job/master/windows_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/)        |
| Linux (CPU)   | GCC 5.5.0  | 3.4.6  |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ecvl/job/master/linux_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/)        |
| Windows (GPU) | VS 15.9.11 | 3.4.6  |  Not available yet        |
| Linux (GPU)   | GCC 5.5.0  | 3.4.6  |  Not available yet        |


## Documentation

The ECVL documentation is available [here](https://deephealthproject.github.io/ecvl/).

## Requirements
- CMake 3.13 or later
- C++ Compiler with C++17 support (e.g. gcc-8 or later, Visual Studio 2017 or later)
- [OpenCV](https://opencv.org) 3.0 or later (modules required: `core`, `imgproc`, `imgcodecs`, `photo`)

### Optional
- [wxWidgets](https://www.wxwidgets.org/), required if `ECVL_BUILD_GUI` flag is enabled
  - OpenGL 3.3 or later, required by the 3D viewer enabled by `ECVL_BUILD_GUI` flag
- [OpenSlide](https://github.com/openslide/openslide), required by `ECVL_WITH_OPENSLIDE` flag

## Installation
Clone and install ECVL with:
```bash
git clone https://github.com/deephealthproject/ecvl.git
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

CMake flags and options:
- `-DECVL_TESTS` (default `ON`): Compiles tests
- `-DECVL_BUILD_EXAMPLES` (default `OFF`): Compiles examples and downloads examples data 
- `-DECVL_DATASET` (default `OFF`): Compiles dataset module
- `-DECVL_BUILD_EDDL` (default `ON`): Compiles eddl integration module (it automatically enables `ECVL_DATASET` option)
- `-DECVL_BUILD_GUI` (default `OFF`): Compiles GUI module
- `-DECVL_WITH_OPENGL` (default `OFF`): Enables 3D GUI functionalities
- `-DECVL_WITH_DICOM` (default `OFF`): Enables DICOM format support
- `-DECVL_WITH_OPENSLIDE` (default `OFF`): Enables OpenSlide whole-slide image support

#### ECVL installation example
ECVL installation with all options enabled and required libraries installed in "non-standard" system directories:
```bash
git clone https://github.com/deephealthproject/ecvl.git
mkdir build && cd build
cmake \
  -DECVL_BUILD_EXAMPLES=ON \
  -DECVL_BUILD_EDDL=ON \
  -DECVL_DATASET=ON \
  -DECVL_BUILD_GUI=ON \
  -DECVL_WITH_OPENGL=ON \
  -DECVL_WITH_DICOM=ON \
  -DECVL_WITH_OPENSLIDE=ON \
  -DCMAKE_INSTALL_PREFIX=install \
  -DOpenCV_DIR=/home/<user>/opencv/build \
  -Deddl_DIR=/home/<user>/eddl/build/cmake \
  -DOPENSLIDE_INCLUDE_DIRECTORIES=/home/<user>/openslide_src/include/openslide \
  -DOPENSLIDE_LIBRARIES=/home/<user>/openslide_src/lib/libopenslide.so \
  -DwxWidgets_CONFIG_EXECUTABLE=/home/<user>/wxWidgets/build/install/bin/wx-config ..
make -j$(nproc)
make install
```

## ImageWatch plugin for Microsoft Visual Studio

An extension of ImageWatch is available to visually inspect ecvl::Image when debugging. In order to use it be sure to install the ImageWatch plugin for Visual Studio and copy and past the file ```tools/ECVL.natvis``` from the GitHub repo into ```C:\Users\<!!username!!>\Documents\Visual Studio 2017\Visualizers```

## ECVL Development Status

ECVL development status is available [here](DEVSTAT.md).

## Contributing

Any contribution is really welcome!

## Contributors

Thanks goes to these wonderful people ([emoji key](https://github.com/all-contributors/all-contributors#emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
