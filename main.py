#!usr/bin/python

###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "Main function for Image Alignment"

__date__ = "06/01/2020"
__maintainer__ = "Damanpreet Kaur"
__version__ = 1.2 
###############################################################################

import utils as utils
import pandas as pd
import os
import cv2
import numpy as np
from config import config as cfg
from skimage.io import imread
import argparse
from PIL import Image
idx_palette = np.reshape(np.asarray(cfg.class_color), (-1))
from numpy import genfromtxt

def get_arguments():
     """
         Parse all the command line arguments.
     """
     parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
     parser.add_argument("--img-csv", '-i', type=str, default=None, help="CSV containing paths for RGB/segmented images. For mode 1 - it should have only one image_path in the CSV file.")
     parser.add_argument("--hyper-img", type=str, default=None, help="Chlorophyll matrix")
     parser.add_argument("--rgb-img", '-r', type=str, default=None, help="Chlorophyll matrix")
     parser.add_argument("--ch", type=int, default=2, help="RGB channel to be matched with the hyperspectral. Applicable for mode 1.")
     parser.add_argument("--grid-type", type=int, choices=[1,2], default=2, help="Grid type options. 1 for 16 samples and 2 for 20 samples.")
     parser.add_argument("--mode", type=int, choices=[1,2], default=2, help="Select Mode (1- to compute the Homography matrix, 2- Apply existing Homography matrix)")
     parser.add_argument("--read_hyper", type=bool, default=True, help="Whether to read the hyperspectral excel file. Default=True. This argument is only valid for mode 1.")
     return parser.parse_args()


def main(args):
    # read the images
    try:
        hyper_img_path = args.hyper_img
    except:
        raise Exception("Please provide path the hyperspectral image for alignment.")

    # read the hyperspectral image
    if '.csv' in hyper_img_path:
        hyp_img = genfromtxt(hyper_img_path, delimiter=',')
        hyp_img = np.uint8(hyp_img)

        hyp_img = cv2.rotate(hyp_img, rotateCode=1)         # rotate the hyperspectral image 
        hyp_img = cv2.flip(hyp_img, 1)
    else:
        hyp_img = cv2.imread(hyper_img_path)

    if args.rgb_img:
        rgb_image_path = args.rgb_img

    #read the image paths from csv
    if args.img_csv or args.mode==2:
        try:
            data = pd.read_csv(args.img_csv)
            rgb_images = data['rgb_images']
        except:
            if not args.img_csv:
                raise Exception("Please add the rgb images in a csv, since mode is 2.")
            else:
                raise Exception("Error while processing the csv file.")

    if args.mode == 2:
        # read the hyperspectral image
        if '.csv' in hyper_img_path:
            hyp_img = genfromtxt(hyper_img_path, delimiter=',')
            hyp_img = np.uint8(hyp_img)
        else:
            hyp_img = cv2.imread(hyper_img_path)

        # preprocess the rgb_img based on given hyperspectral image type
        # apply h_matrix to all rgb images provided in csv file
        for rgb_img_path in rgb_images:
            if '.jpg' in rgb_img_path:
                rgb_img = cv2.imread(rgb_img_path)
                ext = 'jpg'
            elif '.png' in rgb_img_path:
                rgb_img = imread(rgb_img_path, pilmode='P')
                ext = 'png'
            else:
                raise Exception("Image file type not supported.")
            
            # Homography matrix
            if args.grid_type == 1:
                h_matrix = cfg.h_matrix_1_segment if ext=='png' else cfg.h_matrix_1
            elif args.grid_type == 2:
                h_matrix = cfg.h_matrix_2_segment if ext=='png' else cfg.h_matrix_2
            else:
                raise Exception("Grid type not supported.")
                
            if len(hyp_img.shape)==2:   height, width = hyp_img.shape
            else: height,width,_ = hyp_img.shape

            prep_rgb_img = utils.preprocess_rgb(rgb_img, height, width, ext)
            warped_rgb = cv2.warpPerspective(np.array(prep_rgb_img), h_matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if ext=='png':
                warped_rgb = Image.fromarray(warped_rgb)
                warped_rgb.putpalette(list(idx_palette))
                warped_rgb.save(rgb_img_path[:-4]+"_processed.png")
                continue
            cv2.imwrite(rgb_img_path[:-4]+"_processed.jpg", warped_rgb)

    elif args.mode == 1:

        # create the output directory for the image
        DIR_NAME = '.'.join(rgb_image_path.split('/')[-1].split('.')[:-1])
        directory_path = os.path.join('output', DIR_NAME)
        if not os.path.exists('output'):
            os.mkdir('output')

        # call function to read hyper data and preprocess rgb images
        '''
        sheet_number = None if reading all sheets of hyper_img workbook
        else, set sheet_number = the specific sheet_number
        '''
        if(args.read_hyper is True):
            rgb_img, hyp_img = utils.preprocess_hyper_and_rgb(\
            hyp_img, rgb_image_path, directory_path, sheet_number=2)
        else:
            #read images
            rgb_img = cv2.imread(rgb_image_path)
            hyp_img = cv2.imread(hyper_img_path)
        
        # save hyperspectral image
        cv2.imwrite("hyp_img.jpg", hyp_img)

        # align the images and get the results
        # open new_utils to tune the tunable parameters for better homography matrix
        align_img, unalign_img, warped_rgb, homography = utils.align_image(hyp_img, rgb_img, args.ch)

        # Print the computed homography
        print("Homography : \n", homography)
        print()
        print("Saving aligned image at:");
        print(directory_path)

        # Write transformed rgb, aligned, and unaligned images to disk.
        warped_name = r"_transformed_rgb.jpg"
        cv2.imwrite(directory_path+warped_name, warped_rgb)
        align_name = r"_aligned.jpg"
        cv2.imwrite(directory_path+align_name, align_img)
        unalign_name = r"_unaligned.jpg"
        cv2.imwrite(directory_path+unalign_name, unalign_img)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
