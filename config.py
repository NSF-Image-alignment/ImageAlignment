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
	[[-1.01958602e-03,  1.10118349e+00, -1.49211595e+02],
	 [-1.12916699e+00,  7.39082129e-03,  1.65763753e+03],
	 [-6.05997265e-07,  4.36632118e-06,  1.00000000e+00]]
 )

config.h_matrix_2_segment = np.array(
	 [[-3.53909009e-03,  1.09681395e+00, -1.46821608e+02],
	 [-1.12786210e+00, 5.77396115e-03,  1.65486724e+03],
	 [-3.82910853e-06,  2.51942827e-06,  1.00000000e+00]]
 )




config.class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]