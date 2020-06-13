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
	[[ 1.01985640e+00, -3.81582748e-04, -9.10352679e+01],
 [-1.62222165e-03,  1.21362182e+00, -1.48436565e+02],
 [-9.39854757e-07,  3.90843258e-06,  1.00000000e+00]]
 )

config.h_matrix_2_segment = np.array(
[[ 1.01572840e+00, -6.70811737e-04, -8.95515041e+01],
 [-2.69126884e-03,  1.20968370e+00, -1.46584455e+02],
 [-2.04241689e-06,  1.04003375e-06,  1.00000000e+00]]
)



config.class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]