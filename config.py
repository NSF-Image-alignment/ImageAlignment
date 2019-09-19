from easydict import EasyDict as edict
import numpy as np

config = edict()

# crop dimensions for hyperspectral image
config.hyperspec_cropdims = (251, 1830, 330, 1157)
# crop dimensions for rgb image
config.rgb_cropdims = (94, 77, 507, 450)
# homography matrix for warping the image 
config.h_matrix = np.array([[ .986259758,.0329835641, -1.69711514], [ .00100061611, .798943245, 1.79183846e+02], [ 3.74790798e-06, 3.13911362e-05, 1.00000000]])