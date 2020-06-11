#!usr/bin/python
###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "configurations for the homography matrices for Image Alignment"
__date__ = "11/12/2019"
__version__ = 1.0
###############################################################################

from easydict import EasyDict as edict
import numpy as np
config = edict()

#grid_type =1: 16 samples
#grid_type =2: 20 samples

h_ma1 = (
[[ 1.20929054e+00,  9.35797442e-03, -1.49528686e+02],
[-1.63188868e-02,  1.03924034e+00, -1.54596767e+02],
[-2.07130119e-05,  7.15784176e-06,  1.00000000e+00]]
)

#homography matrix for warping the image with grid_type 1
config.h_matrix_1 = np.array(h_ma1)


h_ma2 = (
	[[-8.04829166e-01,  5.59771056e-04,  1.28025245e+03],
 [ 2.21996781e-03, -1.01937778e+00,  1.50154600e+03],
 [-1.82490603e-06,  1.50002751e-06,  1.00000000e+00]]
 )
 

#homography matrix for warping the image with grid_type 2
config.h_matrix_2 = np.array(h_ma2)


config.class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]