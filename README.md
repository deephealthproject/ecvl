
![ECVL](doc/logo/DEEPHEALTH_doxygen_logo_reduced.png)
# ECVL - European Computer Vision Library 

| System  |  Compiler  | OpenCV | Status | 
|:-------:|:----------:|:------:|:------:|
| Windows (CPU) | VS 15.9.11 | 3.4.6  |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ecvl/job/master/windows_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/)        |
| Linux (CPU)   | GCC 5.5.0  | 3.4.6  |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ecvl/job/master/linux_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ecvl/job/master/)        |
| Windows (GPU) | VS 15.9.11 | 3.4.6  |  Not available yet        |
| Linux (GPU)   | GCC 5.5.0  | 3.4.6  |  Not available yet        |


## Documentation

The ECVL documentation is available online [here](http://imagelab.ing.unimore.it/ecvl/). It is automatically updated at each commit/push to the master branch.

## Requirements
- CMake 3.13 or later
- C++ Compiler with C++17 support (e.g. gcc-8 or later, Visual Studio 2017 or later)
- [OpenCV](https://opencv.org) 3.0 or later

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
- `-DECVL_DATASET_PARSER` (default `OFF`): Compiles dataset parser module
- `-DECVL_BUILD_EDDL` (default `ON`): Compiles eddl integration module (it automatically enables `ECVL_DATASET_PARSER` option)
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
  -DECVL_DATASET_PARSER=ON \
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

## ECVL Development Status (Work in progress list)

:heavy_check_mark: Implemented &nbsp; &nbsp; :large_blue_circle: Scheduled/Work in progress &nbsp; &nbsp; :x: Not implemented &nbsp; &nbsp; :no_entry_sign: Not needed

### Image Read
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Standard Formats | :heavy_check_mark: | :x: | :x: |
| NIfTI | :heavy_check_mark: | :x: | :x: |
| DICOM | :heavy_check_mark: | :x: | :x: |
| Whole-slide image <br>(Hamamatsu, Aperio, MIRAX, ...) | :heavy_check_mark: | :x: | :x: |

### Image Write
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Standard Formats | :heavy_check_mark: | :x: | :x: |
| NIfTI | :heavy_check_mark: | :x: | :x: |
| DICOM | :heavy_check_mark: | :x: | :x: |

### Image Arithmetics
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Add | :heavy_check_mark: | :x: | :x: |
| Sub | :heavy_check_mark: | :x: | :x: |
| Mul | :heavy_check_mark: | :x: | :x: |
| Div | :heavy_check_mark: | :x: | :x: |
| Neg | :heavy_check_mark: | :x: | :x: |

### Image Manipulation
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| ChangeColorSpace | :heavy_check_mark: | :x: | :x: |
| Flip | :heavy_check_mark: | :x: | :x: |
| HConcat | :heavy_check_mark: | :x: | :x: |
| Mirror | :heavy_check_mark: | :x: | :x: |
| ResizeDim | :heavy_check_mark: | :x: | :x: |
| ResizeScale | :heavy_check_mark: | :x: | :x: |
| Rotate | :heavy_check_mark: | :x: | :x: |
| RotateFullImage | :heavy_check_mark: | :x: | :x: |
| Stack | :heavy_check_mark: | :x: | :x: |
| VConcat | :heavy_check_mark: | :x: | :x: |

### Pre- Post-processing and Augmentation
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Additive Gaussian Noise | :x: | :x: | :x: |
| Additive Laplace Noise | :heavy_check_mark: | :x: | :x: |
| Additive Poisson Noise | :heavy_check_mark: | :x: | :x: |
| Average Blur | :x: | :x: | :x: |
| Channel Shuffle | :x: | :x: | :x: |
| Coarse Dropout | :heavy_check_mark: | :x: | :x: |
| Connected Components Labeling | :heavy_check_mark: | :x: | :x: |
| Dropout | :x: | :x: | :x: |
| Filter2D | :heavy_check_mark: | :x: | :x: |
| FindContours | :heavy_check_mark: | :x: | :x: |
| Gamma Contrast | :heavy_check_mark: | :x: | :x: |
| Gaussian Blur | :heavy_check_mark: | :x: | :x: |
| Histogram Equalization | :x: | :x: | :x: |
| Impulse Noise | :x: | :x: | :x: |
| Integral Image | :heavy_check_mark: | :x: | :x: |
| Median Blur | :x: | :x: | :x: |
| Non Maxima Suppression | :heavy_check_mark: | :x: | :x: |
| Pepper | :x: | :x: | :x: |
| Salt | :x: | :x: | :x: |
| Salt And Pepper | :x: | :x: | :x: |
| SeparableFilter2D | :heavy_check_mark: | :x: | :x: |
| Threshold | :heavy_check_mark: | :x: | :x: |
