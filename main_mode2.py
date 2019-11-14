#!usr/bin/python
###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "Main function for Image Alignment"
__date__ = "11/12/2019"
__version__ = 1.1
###############################################################################
import new_utils
import pandas as pd
import os
import cv2
import numpy as np
import config as cfg

if __name__ == "__main__":
    #mode =1: calculate homography matrix
    #mode =2: apply homography matrix on rgb images from csv
    mode = 2

    #grid_type = 1: 16 samples 
    #grid_type = 2: 20 samples
    #provide a grid_type number other than 1 and 2 if apply new homography matrix
    grid_type = 2

    if mode == 2:
        #provide the hyper_img of the associated grid_type
        hyper_img_path = r".\output\GWO1_I2.0_F1.9_L80_103450_0_0_0_rgb\full_hyperspec_img.png"
        hyp_img = cv2.imread(hyper_img_path)

        csv_path = r"./test_pipeline/test.csv"
        #read the paths from csv
        data = pd.read_csv(csv_path)
        rgb_images = data['rgb_images']

        #apply the following h_matrix
        if grid_type == 1:
            h_matrix = cfg.h_matrix_1
        elif grid_type == 2:
            h_matrix = cfg.h_matrix_2
        else:
            #provide the homography matrix to be applied
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
        read_hyper = True

        #if True: provide the path to the .xlsx file and raw rgb image
        HYPERSPECTRAL_IMG = r"Alignment_testing/TPB3_I5.0_F1.9_L100_171409_3_0_4-CLS,1.xlsx"
        RGB_IMG = r"Alignment_testing/TPB3_I5.0_F1.9_L100_171409_3_0_4_rgb.jpg"

        # create the output directory for the image
        DIR_NAME = '.'.join(RGB_IMG.split('/')[-1].split('.')[:-1])
        directory_path = os.path.join('output', DIR_NAME)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

        #call function to read hyper data and preprocess rgb images
        '''
        sheet_number = None if reading all sheets of hyper_img workbook
        else, set sheet_number = the specific sheet_number
        '''
        if(read_hyper is True):
            rgb_img, hyp_img = new_utils.preprocess_hyper_and_rgb(\
            HYPERSPECTRAL_IMG, RGB_IMG, directory_path, sheet_number=2)
            hyp_img = cv2.cvtColor(hyp_img, cv2.COLOR_GRAY2BGR)

        #provide the paths to processed hyper_img and rgb_img
        else:
            # load images
            hyper_img_path = r"\output\GWO1_I2.0_F1.9_L80_103450_0_0_0_rgb\full_hyperspec_img.png"
            rgb_img_path =  r"\output\GWO1_I2.0_F1.9_L80_103450_0_0_0_rgb\rgb_prep.png"
            #read images
            rgb_img = cv2.imread(rgb_img_path)
            hyp_img = cv2.imread(hyper_img_path)

        #align the images and get the results
        #open new_utils to tune the tunable parameters for better homography matrix
        align_img, unalign_img, warped_rgb, homography = new_utils.align_image(hyp_img, rgb_img)

# Show the images:
        # cv2.imshow("align_img", align_img)
        # cv2.waitKey(0)
        # cv2.imshow("unalign_img", unalign_img)
        # cv2.waitKey(0)
        # cv2.imshow("warped_rgb", warped_rgb)
        # cv2.waitKey(0)

# Print estimated homography
        print("Homography : \n", homography)
        print()
        print("Saving aligned image at:");
        print(directory_path)

# Write transformed rgb, aligned, and unaligned images to disk.
        warped_name = r"\transformed_rgb.jpg"
        cv2.imwrite(directory_path+warped_name, warped_rgb)
        align_name = r"\aligned.jpg"
        cv2.imwrite(directory_path+align_name, align_img)
        unalign_name = r"\unaligned.jpg"
        cv2.imwrite(directory_path+unalign_name, unalign_img)
