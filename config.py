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
	[[ 1.21256732e+00, -1.70519794e-03, -1.47969186e+02],
	 [-6.85714952e-04,  1.01907901e+00, -9.06223661e+01],
	 [ 3.43777481e-06, -1.14657110e-06,  1.00000000e+00]]
 )

config.h_matrix_2_segment = np.array(
	[[-8.06213346e-01,  1.11629302e-03,  1.28169450e+03],
	 [ 2.46462462e-03, -1.01959082e+00,  1.50215762e+03],
	 [-1.72683736e-06,  2.89790183e-06,  1.00000000e+00]]
 )




config.class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]