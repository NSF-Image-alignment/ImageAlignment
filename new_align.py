#!usr/bin/python
###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "Image Alignment"
__date__ = "10/20/2019"
'''
disclaimer-tracking features code from:
https://stackoverflow.com/questions/45162021/
python-opencv-aligning-and-overlaying-multiple-images-one-after-another
'''
###############################################################################
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from new_utils import read_hyper_data, preprocess_rgb, ImageAlignment

def align_images(hyp_img, rgb_img):

    #preprocess to hsv
    rgb_hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("rgb_hsv", rgb_hsv)
    # cv2.waitKey(0)
    cv2.imwrite("hsv_img.png", rgb_hsv)
    # Convert images to grayscale
    rgb_gray = cv2.cvtColor(rgb_hsv, cv2.COLOR_BGR2GRAY)
    hyp_gray = cv2.cvtColor(hyp_img, cv2.COLOR_BGR2GRAY)
    #best:
    # rgb_thresh = cv2.adaptiveThreshold(rgb_gray,125,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY_INV,21,5)
    # hyp_thresh = cv2.adaptiveThreshold(hyp_gray,125,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY_INV,9,2)

    #BEST:
    rgb_thresh = cv2.adaptiveThreshold(rgb_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,15,5)

    hyp_thresh = cv2.adaptiveThreshold(hyp_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
    cv2.imwrite("rgb_thresh.png", rgb_thresh)
    cv2.imwrite("hyp_thresh.png", hyp_thresh)
    cv2.imshow("rgbthresh", rgb_thresh)
    cv2.waitKey(0)
    cv2.imshow("hypthresh", hyp_thresh)
    cv2.waitKey(0)

    # find the coordinates of good features to track  in prep_rgb_img
    hyp_features = cv2.goodFeaturesToTrack(hyp_thresh, 10000, .0995, 5)
    # find corresponding features in current photo
    rgb_features = np.array([])
    rgb_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(hyp_thresh, rgb_thresh, hyp_features, rgb_features, flags=1)

    print(len(pyr_stati))
    # only add features for which a match was found to the pruned arrays
    good_rgb_features = []
    good_hyp_features = []
    for index, status in enumerate(pyr_stati):
        if status == 1:
            good_rgb_features.append(rgb_features[index])
            good_hyp_features.append(hyp_features[index])

    # convert lists to numpy arrays so they can be passed to opencv function
    rgb_final_features = np.asarray(good_rgb_features)
    hyp_final_features = np.asarray(good_hyp_features)

    # find perspective transformation using the arrays of corresponding points
    h_transformation = cv2.findHomography(rgb_final_features, hyp_final_features, method=cv2.RANSAC, ransacReprojThreshold=1)[0]

    # transform the images and overlay them to see if they align properly
    height, width = rgb_img.shape[:2]
    warped_rgb = cv2.warpPerspective(rgb_img, h_transformation, (width, height))
    cv2.imshow("warped_rgb", warped_rgb)
    cv2.waitKey(0)

    align_img = cv2.addWeighted(warped_rgb, .3, hyp_img, .7, 1)

    # unalign_img = cv2.addWeighted(rgb_img, .3, hyp_img, .7, 1)
    # cv2.imwrite("unalign_img.png", unalign_img)
    return align_img, h_transformation

#This function reads the hyperspectral excel workbook
#and get the correct orientation of the hyper image
#and preprocess hyper and rgb image for alignment
def preprocess_hyper_and_rgb(hs_img, rgb_image,directory_path, sheet_number, grid_type):
    #read hyperspectral data from excel workbook
    hyper_img,hyper_shape = read_hyper_data(hs_img, directory_path,\
                                sheet_number, grid_type, choose_best = False)

    #rescale image to (600,600)
    rgb_img = cv2.imread(RGB_IMG)
    rgb_prep = preprocess_rgb(rgb_img, hyper_img)
    cv2.imwrite(directory_path+"/rgb_prep.png", rgb_prep)
    print("------------Preprocess is saved and finished.-------------------")
    return rgb_prep, hyper_img


#run from command line:
#python new_align.py
if __name__ == '__main__':
    #set the following to True:
    #if need to read data from .xlsx files and preprocess imgs:
    #set to False:
    #if hyper image is already obtained and preprocessed
    read_hyper = False

    #following are the paths to the input files of raw hyper data and rgb image
    HYPERSPECTRAL_FILE = r'\\depot.engr.oregonstate.edu\users\lijiay\Windows.Documents\Desktop\ImageAlignment\Alignment_testing\TPB2_I5.0_F1.9_L100_170342_14_2_2-CLS,1.xlsx'
    RGB_IMG = './Alignment_testing/TPB2_I5.0_F1.9_L100_170342_14_2_2_rgb.jpg'
    # create the output directory
    DIR_NAME = '.'.join(RGB_IMG.split('/')[-1].split('.')[:-1])
    directory_path = os.path.join('output', DIR_NAME)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    if(read_hyper is True):
        rgb_img, hyp_img = preprocess_hyper_and_rgb(HYPERSPECTRAL_FILE,\
                                    RGB_IMG, directory_path, sheet_number =2, grid_type=2)
    else:
        img_path = r"\\depot.engr.oregonstate.edu\users\lijiay\Windows.Documents\Desktop\ImageAlignment\output"
        image_name = r"\VA1_I5.0_F1.9_L100_cyan_163545_6_1_5_rgb"
        hyper_name = r"\full_hyperspec_img.png"
        rgb_name = r"\rgb_prep.png"
        # load images
        hyper_img_path = img_path + image_name + hyper_name
        rgb_img_path =  img_path + image_name + rgb_name
        #read images
        rgb_img = cv2.imread(rgb_img_path)
        hyp_img = cv2.imread(hyper_img_path)
        output_path = img_path + image_name

    print("___________________Aligning images______________________")
    # ALigned image will be resotred in align_img
    # The estimated homography will be stored in h.
    #align_img, h = align_images(hyp_img,rgb_img)
    #cv2.imshow("align", align_img)
    #cv2.waitKey(0)

    # exit(0)

    # # ALigned image by applying homography will be resotred in align_img
    align = ImageAlignment(grid_type = 2)
    align_img, h = align.warp_image(hyp_img, rgb_img, directory_path)
    cv2.imshow("align", align_img)
    cv2.waitKey(0)

    print("_________________Saved aligned image at: ___________________")
    if read_hyper is True:
        align_name = r"\aligned.jpg"
        print(directory_path+align_name)
        cv2.imwrite(directory_path + align_name, align_img)
    else:
        align_name = r"\aligned.jpg"
        print(output_path+align_name)
        cv2.imwrite(output_path + align_name, align_img)

    # Print estimated homography
    print("Homography : \n",  h)

#TODO: check error:
#cv2.error: OpenCV(4.1.1) C:\projects\opencv-python\opencv\modules\core\src\arithm.cpp:663: error: (-209:Sizes of input arguments do not match) The operation is neither 'array op array' (where arrays have the same size and t
#he same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'

#Below are testing homographies:
#good:
# [[ 1.04975026e+00 -6.16853150e-03 -8.67832588e+01]
# [-6.69922812e-02  9.75009953e-01 -1.06551893e+02]
# [-1.02389367e-04 -3.22661322e-06  1.00000000e+00]]

# [[ 1.04956065e+00 -6.12782508e-03 -8.64834382e+01]
# [-6.61971277e-02  9.75893293e-01 -1.07517531e+02]
# [-1.02300676e-04 -2.86749861e-06  1.00000000e+00]]

# [[ 1.06634268e+00 -7.68582899e-03 -9.05334812e+01]
# [-2.98486972e-02  9.59922655e-01 -1.24472116e+02]
# [-8.91817312e-05 -1.43342462e-05  1.00000000e+00]]

# [[ 1.22304313e+00 -4.67999420e-03 -1.46693835e+02]
# [-2.19954996e-02  1.05917254e+00 -1.55884387e+02]
# [ 4.03600545e-05 -1.55941653e-05  1.00000000e+00]]


###best! grid 1
# [[ 1.18755506e+00  1.12908991e-02 -1.49696027e+02]
#  [-4.06464940e-02  1.04059629e+00 -1.44030779e+02]
#  [-3.26477331e-05  7.28288881e-06  1.00000000e+00]]

# [[ 1.20929054e+00  9.35797442e-03 -1.49528686e+02]
#  [-1.63188868e-02  1.03924034e+00 -1.54596767e+02]
#  [-2.07130119e-05  7.15784176e-06  1.00000000e+00]]


#best grid 2:
 # [[ 1.24176399e+00 -1.45553802e-02 -1.43366108e+02]
 # [ 3.63842479e-02  1.03508395e+00 -1.40048578e+02]
 # [ 2.19053828e-05 -2.38947223e-05  1.00000000e+00]]

 # [[ 1.32671276e+00 -1.94890706e-02 -1.59491327e+02]
 # [ 4.69965080e-02  1.10000029e+00 -1.51260785e+02]
 # [ 1.02480100e-04 -1.50408834e-05  1.00000000e+00]]





#!usr/bin/python
###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "Main function for Image Alignment"
__date__ = "10/20/2019"
__version__ = 1.0
###############################################################################
import argparse
import new_utils
import pickle
import pandas as pd
import os
import cv2
import numpy as np
import config as cfg

if __name__ == "__main__":
    #mode =1: calculate homography matrix
    #mode =2: apply homography matrix on rgg images from csv
    mode = 1

    #grid_type =1: 16 samples
    #grid_type =2: 20 samples
    grid_type = 1

    if mode == 2:
        #provide the hyper_img of the associated grid_type
        hyper_img_path = r"\\depot.engr.oregonstate.edu\users\lijiay\Windows.Documents\Desktop\ImageAlignment\output\GWO1_I2.0_F1.9_L80_103450_0_0_0_rgb\full_hyperspec_img.png"
        hyp_img = cv2.imread(hyper_img_path)

        csv_path = r"\\depot.engr.oregonstate.edu\users\lijiay\Windows.Documents\Desktop\ImageAlignment\test_pipeline\test.csv"
        #need to read the files from csv
        data = pd.read_csv(csv_path)
        rgb_images = data['rgb_images']

        #apply the following h_matrix
        # h_matrix = cfg.h_matrix_1
        h_matrix = np.array(
        [[ 1.20929054e+00,  9.35797442e-03, -1.49528686e+02],
        [-1.63188868e-02,  1.03924034e+00, -1.54596767e+02],
        [-2.07130119e-05,  7.15784176e-06,  1.00000000e+00]])

        #preprocess the rgb_img based on given hyperspectral image type
        #apply h_matrix to all rgb images provided in csv file
        for rgb_img_path in rgb_images:
            rgb_img = cv2.imread(rgb_img_path)
            prep_rgb_img = new_utils.preprocess_rgb(rgb_img, hyp_img)
            height, width = prep_rgb_img.shape[:2]
            warped_rgb = cv2.warpPerspective(prep_rgb_img, h_matrix, (width, height))
            cv2.imwrite(rgb_img_path[:-4]+"_processed.jpg", warped_rgb)

    if mode == 1:
        #True:if need to read data from .xlsx files for hyperspectral images:
        #False:if hyper and rgb images are already obtained and preprocessed
        read_hyper = False

        #if True: provide the path to the .xlsx file and raw rgb image
        HYPERSPECTRAL_IMG = './Alignment_testing/GWO1_I2.0_F1.9_L80_103704_3_0_4.xlsx'
        RGB_IMG = './Alignment_testing/GWO1_I2.0_F1.9_L80_103704_3_0_4_rgb.jpg'

        # create the output directory for the image
        DIR_NAME = '.'.join(RGB_IMG.split('/')[-1].split('.')[:-1])
        directory_path = os.path.join('output', DIR_NAME)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        #call function to read hyper data and preprocess rgb images
        if(read_hyper is True):
            rgb_img, hyp_img = new_utils.preprocess_hyper_and_rgb(\
            HYPERSPECTRAL_IMG, RGB_IMG, directory_path, sheet_number=2, grid_type=1)

        #provide the paths to processed hyper_img and rgb_img
        else:
            img_path = r"\\depot.engr.oregonstate.edu\users\lijiay\Windows.Documents\Desktop\ImageAlignment\output"
            image_name = r"\GWO1_I2.0_F1.9_L80_103450_0_0_0_rgb"
            hyper_name = r"\full_hyperspec_img.png"
            rgb_name = r"\rgb_prep.png"
            # load images
            hyper_img_path = img_path + image_name + hyper_name
            rgb_img_path =  img_path + image_name + rgb_name
            # # load images
            # hyper_img_path = r"path to hyper_img"
            # rgb_img_path =  r"path to rgb"
            #read images
            rgb_img = cv2.imread(rgb_img_path)
            hyp_img = cv2.imread(hyper_img_path)
            # cv2.imshow("rgb_img", rgb_img)
            # cv2.waitKey(0)

        #align the images and get the results
        align_img, unalign_img, warped_rgb, homography = new_utils.align_image(hyp_img, rgb_img)

# Show the images:
        cv2.imshow("align_img", align_img)
        cv2.waitKey(0)
        cv2.imshow("unalign_img", unalign_img)
        cv2.waitKey(0)
        cv2.imshow("warped_rgb", warped_rgb)
        cv2.waitKey(0)

# Print estimated homography
        print("Homography : \n", homography)
        print()
        print("_________________Saving aligned image at: ___________________");
        print(directory_path)
        print()

# Write transformed rgb, aligned, and unaligned images to disk.
        # warped_name = r"\transformed_rgb.jpg"
        # cv2.imwrite(directory_path+warped_name, warped_rgb)
        # align_name = r"\aligned.jpg"
        # cv2.imwrite(directory_path+align_name, align_img)
        # unalign_name = r"\unaligned.jpg"
        # cv2.imwrite(directory_path+unalign_name, unalign_img)
