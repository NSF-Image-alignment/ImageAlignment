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
[[ 1.20262572e+00, -4.79663041e-03, -1.43459905e+02],
[-4.59787369e-03,  1.01879219e+00, -9.79267852e+01],
[ 2.44805399e-06, -1.11496491e-05,  1.00000000e+00]]
)

#homography matrix for warping the image with grid_type 2
config.h_matrix_2 = np.array(h_ma2)
