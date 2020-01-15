
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
- CMake 3.13 or later;
- C++ Compiler with C++17 support;
- OpenCV 3.0 or later (https://opencv.org/).

### Optional
- wxWidgets (https://www.wxwidgets.org/), required by the ECVL GUI module;
- OpenGL 3.3 or later, required by the 3D viewer.

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
| Whole-slide image <br>(Hamamatsu, Aperio, MIRAX, ...) | :large_blue_circle: | :x: | :x: |

### Image Arithmetics
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Add | :heavy_check_mark: | :x: | :x: |
| Sub | :heavy_check_mark: | :x: | :x: |
| Mul | :heavy_check_mark: | :x: | :x: |
| Div | :heavy_check_mark: | :x: | :x: |
| Neg | :heavy_check_mark: | :x: | :x: |


### Pre- Post-processing and Augmentation
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Channel Shuffle | :x: | :x: | :x: |
| Add | :heavy_check_mark: | :x: | :x: |
| Additive Gaussian Noise | :x: | :x: | :x: |
| Additive Laplace Noise | :heavy_check_mark: | :x: | :x: |
| Additive Poisson Noise | :x: | :x: | :x: |
| Impulse Noise | :x: | :x: | :x: |
| Mul | :x: | :x: | :x: |
| Dropout | :x: | :x: | :x: |
| Coarse Dropout | :heavy_check_mark: | :x: | :x: |
| Salt And Pepper | :x: | :x: | :x: |
| Salt | :x: | :x: | :x: |
| Pepper | :x: | :x: | :x: |
| Invert | :x: | :x: | :x: |
| Gaussian Blur | :heavy_check_mark: | :x: | :x: |
| Average Blur | :x: | :x: | :x: |
| Median Blur | :x: | :x: | :x: |
| Histogram Equalization | :x: | :x: | :x: |
| Flip | :heavy_check_mark: | :x: | :x: |
| Gamma Contrast | :heavy_check_mark: | :x: | :x: |
