from easydict import EasyDict as edict
import numpy as np

config = edict()

# crop dimensions for hyperspectral image
# config.hyperspec_cropdims_1 = (251, 1830, 330, 1157)
# config.hyperspec_cropdims_2 = (108, 1348, 240, 1218)
config.hyperspec_cropdims_1 = (30, 1380, 80, 2270)
config.hyperspec_cropdims_2 = (108, 1348, 240, 1218)

# crop dimensions for rgb image
# config.rgb_cropdims_1 = (94, 77, 507, 450)
# config.rgb_cropdims_2 = (48, 136, 510, 438)
# config.rgb_cropdims_1 = (4, 7, 597, 520)
config.rgb_cropdims_2 = (48, 136, 510, 438)

# homography matrix for warping the image 
config.h_matrix_1 = np.array([[ .986259758,.0329835641, -1.69711514], [ .00100061611, .798943245, 1.79183846e+02], [ 3.74790798e-06, 3.13911362e-05, 1.00000000]])
config.h_matrix_2 = np.array([[ 1.03121360e+00, 5.80620575e-02, 1.01817998e+01], [1.32913410e-04, 1.18934057e+00, -5.75134351e+01], [8.74434726e-06, 4.77127812e-05, 1.00000000e+00]])
