
![ECVL](doc/logo/DEEPHEALTH_doxygen_logo_reduced.png)
# ECVL - European Computer Vision Library

| System  |  Compiler  | OpenCV | Status | 
|:-------:|:----------:|:------:|:------:|
| Windows | VS 15.9.11 | 3.4.6  |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ECVL/job/master/windows_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ECVL/job/master/)        |
| Linux   | GCC 5.5.0  | 3.4.6  |[![Build Status](https://jenkins-master-deephealth-unix01.ing.unimore.it/badge/job/DeepHealth/job/ECVL/job/master/linux_end?)](https://jenkins-master-deephealth-unix01.ing.unimore.it/job/DeepHealth/job/ECVL/job/master/)        |

## Documentation

The ECVL documentation is available online [here](http://imagelab.ing.unimore.it/ecvl/) or in the doc folder of the GitHub repository.

## Requirements

- C++ Compiler with C++11 support
- OpenCV 3.0 or later

## ImageWatch plugin for Microsof Visual Studio

A basic version of ImageWatch is available to visually inspect ecvl::Image when debugging. In order to use it be sure to install the ImageWatch plugin for Visual Studio and copy and past the file ```tools/ECVL.natvis``` from the GitHub repo into ```C:\Users\<!!username!!>\Documents\Visual Studio 2017\Visualizers```
