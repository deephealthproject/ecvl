
# ECVL Development Status

:heavy_check_mark: Implemented &nbsp; &nbsp; :large_blue_circle: Scheduled/Work in progress &nbsp; &nbsp; :x: Not implemented &nbsp; &nbsp; :no_entry_sign: Not needed

## Image Read
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Standard Formats | :heavy_check_mark: | :x: | :x: |
| NIfTI | :heavy_check_mark: | :x: | :x: |
| DICOM | :heavy_check_mark: | :x: | :x: |
| Whole-slide image <br>(Hamamatsu, Aperio, MIRAX, ...) | :heavy_check_mark: | :x: | :x: |

## Image Write
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Standard Formats | :heavy_check_mark: | :x: | :x: |
| NIfTI | :heavy_check_mark: | :x: | :x: |
| DICOM | :heavy_check_mark: | :x: | :x: |

## Image Arithmetics
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Add | :heavy_check_mark: | :x: | :x: |
| Sub | :heavy_check_mark: | :x: | :x: |
| Mul | :heavy_check_mark: | :x: | :x: |
| Div | :heavy_check_mark: | :x: | :x: |
| Neg | :heavy_check_mark: | :x: | :x: |

## Image Manipulation
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
| Transpose | :heavy_check_mark: | :x: | :x: |
| VConcat | :heavy_check_mark: | :x: | :x: |
| SetTo | :heavy_check_mark: | :x: | :x: |
| DropColorChannel | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

## Pre- Post-processing and Augmentation
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Additive Gaussian Noise | :x: | :x: | :x: |
| Additive Laplace Noise | :heavy_check_mark: | :x: | :x: |
| Additive Poisson Noise | :heavy_check_mark: | :x: | :x: |
| Average Blur | :x: | :x: | :x: |
| Channel Shuffle | :x: | :x: | :x: |
| Coarse Dropout | :heavy_check_mark: | :x: | :x: |
| Connected Components Labeling | :heavy_check_mark: | :x: | :x: |
| Elastic Transform | :heavy_check_mark: | :x: | :x: |
| Filter2D | :heavy_check_mark: | :x: | :x: |
| FindContours | :heavy_check_mark: | :x: | :x: |
| Gamma Contrast | :heavy_check_mark: | :x: | :x: |
| Gaussian Blur | :heavy_check_mark: | :x: | :x: |
| Grid Distorsion | :heavy_check_mark: | :x: | :x: |
| Histogram Equalization | :x: | :x: | :x: |
| Impulse Noise | :x: | :x: | :x: |
| Integral Image | :heavy_check_mark: | :x: | :x: |
| Median Blur | :x: | :x: | :x: |
| Non Maxima Suppression | :heavy_check_mark: | :x: | :x: |
| OpticalDistortion | :heavy_check_mark: | :x: | :x: |
| Pepper | :heavy_check_mark: | :x: | :x: |
| Salt | :heavy_check_mark: | :x: | :x: |
| Salt And Pepper | :heavy_check_mark: | :x: | :x: |
| SeparableFilter2D | :heavy_check_mark: | :x: | :x: |
| SliceTimingCorrection | :heavy_check_mark: | :x: | :x: |
| Threshold | :heavy_check_mark: | :x: | :x: |
| MultiThreshold | :heavy_check_mark: | :x: | :x: |

## Property Calculation 
| Functionality | CPU | GPU | FPGA |
|--|--|--|--|
| Otsu | :heavy_check_mark: | :x: | :x: |
| Otsu Multi Threshold | :heavy_check_mark: | :x: | :x: |
| Moments | :heavy_check_mark: | :x: | :x: |
| CentralMoments | :heavy_check_mark: | :x: | :x: |
