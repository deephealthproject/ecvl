
![ECVL](doc/logo/DEEPHEALTH_doxygen_logo_reduced.png)
# ECVL - European Computer Vision Library 
[![release](https://img.shields.io/github/v/release/deephealthproject/ecvl)](https://github.com/deephealthproject/ecvl/releases/latest/)
[![docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat)](https://deephealthproject.github.io/ecvl/)
[![build](https://travis-ci.com/deephealthproject/ecvl.svg?branch=master)](#CI)
[![cobertura](https://img.shields.io/jenkins/coverage/cobertura?jobUrl=https%3A%2F%2Fjenkins-master-deephealth-unix01.ing.unimore.it%2Fjob%2FDeepHealth%2Fjob%2Fecvl%2Fjob%2Fmaster%2F&label=cobertura)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/cobertura/)
[![codecov](https://codecov.io/gh/deephealthproject/ecvl/branch/master/graph/badge.svg)](https://codecov.io/gh/deephealthproject/ecvl)
[![license](https://img.shields.io/github/license/deephealthproject/ecvl)](https://github.com/deephealthproject/ecvl/blob/master/LICENSE)<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Documentation

The ECVL documentation is available [here](https://deephealthproject.github.io/ecvl/).

## Requirements
- CMake 3.13 or later
- C++ Compiler with C++17 support (e.g. GCC 6 or later, Clang 5.0 or later, Visual Studio 2017 or later)
- [OpenCV](https://opencv.org) 3.0 or later (modules required: `core`, `imgproc`, `imgcodecs`, `photo`, [`calib3d` since OpenCV 4.0 only])

### Optional
- [EDDL](https://github.com/deephealthproject/eddl), European Distributed Deep Learning Library (`ECVL_BUILD_EDDL` flag)
- [wxWidgets](https://www.wxwidgets.org/), required if `ECVL_BUILD_GUI` flag is enabled
  - OpenGL 3.3 or later, required by the 3D viewer enabled by `ECVL_BUILD_GUI` flag
- [OpenSlide](https://github.com/openslide/openslide), required with `ECVL_WITH_OPENSLIDE` flag

#### EDDL Version Compatibility
|   ECVL  |  EDDL  | 
|:-------:|:------:|
| 0.2.3 | 0.6.0 |
| 0.2.2 | 0.6.0 |
| 0.2.1 | 0.4.3 |
| 0.2.0 | 0.4.3 |
| 0.1.1 | 0.4.2 |
| 0.1.0 | 0.3.1 |

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

ECVL development status is available [here](PROGRESS.md).

## Contributing

Any contribution is really welcome!

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/CostantinoGrana"><img src="https://avatars2.githubusercontent.com/u/18437151?v=1" width="100px;" alt=""/><br /><sub><b>Costantino Grana</b></sub></a><br /><a href="https://github.com/deephealthproject/ecvl/commits?author=CostantinoGrana" title="Code">💻</a> <a href="#ideas-CostantinoGrana" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-CostantinoGrana" title="Tools">🔧</a></td>
    <td align="center"><a href="https://github.com/prittt"><img src="https://avatars.githubusercontent.com/u/6863130?v=1" width="100px;" alt=""/><br /><sub><b>Federico Bolelli</b></sub></a><br /><a href="https://github.com/deephealthproject/ecvl/commits?author=prittt" title="Code">💻</a> <a href="https://github.com/deephealthproject/ecvl/commits?author=prittt" title="Documentation">📖</a> <a href="#tool-prittt" title="Tools">🔧</a></td>
    <td align="center"><a href="https://github.com/MicheleCancilla"><img src="https://avatars2.githubusercontent.com/u/22983812?v=1" width="100px;" alt=""/><br /><sub><b>Michele Cancilla</b></sub></a><br /><a href="https://github.com/deephealthproject/ecvl/commits?author=MicheleCancilla" title="Code">💻</a> <a href="https://github.com/deephealthproject/ecvl/pulls?q=is%3Apr+reviewed-by%3AMicheleCancilla" title="Reviewed Pull Requests">👀</a> <a href="#tool-MicheleCancilla" title="Tools">🔧</a> <a href="https://github.com/deephealthproject/ecvl/commits?author=MicheleCancilla" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/lauracanalini"><img src="https://avatars.githubusercontent.com/u/44258837?v=1" width="100px;" alt=""/><br /><sub><b>Laura Canalini</b></sub></a><br /><a href="https://github.com/deephealthproject/ecvl/commits?author=lauracanalini" title="Code">💻</a> <a href="https://github.com/deephealthproject/ecvl/pulls?q=is%3Apr+reviewed-by%3Alauracanalini" title="Reviewed Pull Requests">👀</a> <a href="#example-lauracanalini" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/stal12"><img src="https://avatars2.githubusercontent.com/u/34423515?v=1" width="100px;" alt=""/><br /><sub><b>Stefano Allegretti</b></sub></a><br /><a href="https://github.com/deephealthproject/ecvl/commits?author=stal12" title="Code">💻</a> <a href="#infra-stal12" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#tool-stal12" title="Tools">🔧</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification.
Contributions of any kind are welcome!

## <a name="CI">Continuous integration (CPU)</a>

### Windows

|   OS    |  Compiler  | OpenCV | EDDL | Infrastructure | Status | 
|:-------:|:----------:|:------:|:------:|:------:|:------:|
| Windows 10 1903 | VS 15.9.11 | 3.4.11 | 0.7.0 | Jenkins |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ecvl/job/master/windows_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/)        |

### Linux

|   OS    |  Compiler  | OpenCV | EDDL |Infrastructure | Status | 
|:-------:|:----------:|:------:|:------:|:------:|:------:|
| Ubuntu 18.04.3  | GCC 8.4.0  | 3.4.6  | 0.6.0 | Jenkins |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ecvl/job/master/linux_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/)        |
| Ubuntu 18.04.4  | GCC 6.5.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/1?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | GCC 7.5.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/2?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | GCC 8.4.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/3?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | GCC 9.3.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/4?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | GCC 10.1.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/5?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | Clang 5.0.2  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/6?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | Clang 6.0.1  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/7?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | Clang 7.1.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/8?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | Clang 8.0.1  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/9?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | Clang 9.0.0  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/10?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)        |
| Ubuntu 18.04.4  | Clang 10.0.1  | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/11?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)       |

### MacOS

|   OS    |  Compiler  | OpenCV | EDDL |Infrastructure | Status | 
|:-------:|:----------:|:------:|:------:|:------:|:------:|
| MacOSX 10.15.4 | Apple Clang 11.0.3 | 3.4.11  | 0.7.0 | Travis CI |[![Build Status](https://travis-matrix-badges.herokuapp.com/repos/deephealthproject/ecvl/branches/master/11?use_travis_com=true)](https://travis-ci.com/github/deephealthproject/ecvl)       |

## <a name="CI">Continuous integration (GPU)</a> 

### Windows

|   OS    |  Compiler  | OpenCV | EDDL |Infrastructure | Status | 
|:-------:|:----------:|:------:|:------:|:------:|:------:|
| Windows 10 1903 | VS 16.2.0 | - | - | Jenkins |  Not available yet        |

### Linux

|   OS    |  Compiler  | OpenCV | EDDL | Infrastructure | Status | 
|:-------:|:----------:|:------:|:------:|:------:|:------:|
| Linux (GPU)   | GCC 8.4.0  | 3.4.6  | - | Jenkins | Not available yet        |

