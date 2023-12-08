# Canny Edge Detector - MATLAB Implementation

## Overview

This MATLAB implementation of the Canny Edge Detector algorithm is designed to accurately segment normalized pixel magnitudes through a standardized Canny Edge Detection methodology. The algorithm leverages hysteresis thresholding to classify pixel magnitudes and achieve robust edge segmentation.

## Usage

To use the implementation, the following input parameters can be adjusted:

- `imNum`: Specifies the image number and is currently set to a default value of 1.
- `ImDir`: Specifies the image directory and is set to a default value of 'Images/'. Adjust these parameters based on your working directory and the desired image.

## Execution

The algorithm comprises a multi-fold approach, executing four major sub-functions to achieve the desired results. The key steps involve image preprocessing, gradient calculation, non-maximum suppression, and hysteresis thresholding. These steps collectively contribute to the accurate segmentation of edges in the input image.

## Results

The output of the algorithm includes both visualizations of predicted boundaries and quantitative measures of precision and recall accuracy. The predicted boundaries are displayed graphically, providing an intuitive representation of the algorithm's segmentation performance.

## Customization

Users can adapt the algorithm to different working directories and images by modifying the `imNum` and `ImDir` parameters. Additionally, fine-tuning of threshold values can be performed for optimal edge detection based on specific image characteristics.

## Citation

If you find this implementation useful for your work, please consider citing it using the following BibTeX entry:

```latex
@article{F-E-H-P:2023,
  title={Canny Edge Detector - MATLAB Implementation},
  author={E.H.P, Felix},
  journal={GitHub Repository},
  year={2023},
  url={https://github.com/felix-e-h-p/canny-edge-detector},
}
