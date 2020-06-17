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

h_ma1_segment = (
[[ 1.20929054e+00,  9.35797442e-03, -1.49528686e+02],
[-1.63188868e-02,  1.03924034e+00, -1.54596767e+02],
[-2.07130119e-05,  7.15784176e-06,  1.00000000e+00]]
)

#homography matrix for warping the image with grid_type 1
config.h_matrix_1 = np.array(h_ma1)
config.h_matrix_1_segment = np.array(h_ma1_segment)


config.h_matrix_2 = np.array(
	 [[ 1.21361802e+00, -1.46332349e-03, -1.48482039e+02],
	 [-3.72407431e-04,  1.02000764e+00, -9.11057685e+01],
	 [ 3.85154964e-06, -7.88660275e-07,  1.00000000e+00]]
 )

config.h_matrix_2_segment = np.array(
	[[ 1.20963756e+00, -2.70444074e-03, -1.46561734e+02],
	 [-6.84428793e-04,  1.01569306e+00, -8.95343658e+01],
	 [ 1.01967586e-06, -2.05846087e-06,  1.00000000e+00]]
 )




config.class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]