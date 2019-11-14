#!usr/bin/python
###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "Utility functions for Image Alignment"
__date__ = "10/20/2019"
'''
disclaimer-tracking features code from:
https://stackoverflow.com/questions/45162021/
python-opencv-aligning-and-overlaying-multiple-images-one-after-another
'''
###############################################################################
import os
from xlrd import open_workbook, XL_CELL_NUMBER
import numpy as np
import cv2
from config import config as cfg
import time
from skimage import measure

def sheet_to_array(filename, sheet_number):
    """
    source - https://gist.github.com/gamesbook/b309cd01b6f97a675a36

    Return a floating-point numpy array from sheet in an Excel spreadsheet.
    Notes:
    0. The array is empty by default; and any non-numeric data in the sheet will
       be skipped.
    1. If first_col is 0 and last_col is None, then all columns will be used,
    2. If header is True, only one header row is assumed.
    3. All rows are loaded.
    """

    try:
        book = open_workbook(filename, on_demand=True)
    except:
        print("Check if the file is present. If it present remove the quotes around the file name and try again.")
        exit()

    sheet_cnt = book.nsheets
    #if a sheet_number is provided, only read that sheet number
    if sheet_number !=  None:
        hyp_data = []
        # for sheet_number in range(sheet_cnt):
        sheet = book.sheet_by_index(sheet_number)
        # print("original width",sheet.nrows )
        # print("original height",sheet.ncols )
        # rows = sheet.nrows
        rows = 1400

        last_col = sheet.ncols
        data = np.empty([last_col, rows], dtype=int)
        cols = [col for col in range(0, last_col + 1)]

        for row in range(0, rows):
            row_values = sheet.row(row)
            for col, cell in enumerate(row_values):
                if col in cols and cell.ctype == XL_CELL_NUMBER:
                    data[col, row] = cell.value

            hyp_data = data
        return hyp_data, sheet_cnt

    else:
        datas = []
        for sheet_number in range(sheet_cnt):
            sheet = book.sheet_by_index(1)
            # rows = sheet.nrows
            rows = 1400

            last_col = sheet.ncols
            data = np.empty([last_col, rows], dtype=int)

            cols = [col for col in range(0, last_col + 1)]

            for row in range(0, sheet.nrows):
                row_values = sheet.row(row)
                for col, cell in enumerate(row_values):
                    if col in cols and cell.ctype == XL_CELL_NUMBER:
                        data[col, row] = cell.value

                datas.append(data)

        return datas, sheet_cnt

#function to read data from .xslx hyperspectral files
def read_hyper_data(hs_img, directory_path, sheet_number, choose_best = False):
    print("-------Begining to read hyperspectral data----------- ")

    #if a sheet_number is provided, only read that sheet number
    if sheet_number !=  None:
        start_time = time.time()
        data, sheet_cnt = sheet_to_array(hs_img, sheet_number)
        end_time = time.time()
        print("Total time taken during reading count_worksheets: " + str(end_time - start_time))
        #preprocess the hs_img using data read from sheet
        hyp_im = preprocess_hyperdata(data)

        # print(hyp_im.shape)
        # cv2.imshow("align", hyp_im)
        # cv2.waitKey(0)

        print("________Read sheet " + str(sheet_number)+ " of hyperspectral image.__________")
        cv2.imwrite(directory_path+"/full_hyperspec_img.png", hyp_im)
        return hyp_im

    #if NO sheet_number is provided, read the entire workbook
    else:
        start_time = time.time()
        datas, sheet_cnt = sheet_to_array(hs_img, sheet_number)
        end_time = time.time()
        print("Total time taken during reading count_worksheets: " + str(end_time - start_time))
        imgs = []
        entropies =[]
        #iterates through the number of channels
        for i in range(len(datas)):
        #reads data value from the sheet of hs_img
            data = datas[i]
            #preprocess the hs_img using data read from sheet
            hyp_im = preprocess_hyperdata(data)
            #output preprocessed hyp_im to path
            cv2.imwrite(directory_path+"/full_hyperspec_img_"+str(i)+".png", hyp_im)
            #the following steps are for calculating the best hyper_img
            imgs.append(hyp_im)
            #calculate the entropy of image
            entropy = measure.shannon_entropy(hyp_im)
            # print("entropy"+str(i), entropy)
            entropies.append(entropy)

        if choose_best == True:
            #output best preprocessed hs_img to path
            best_hyper_index = np.argmax(entropies)
            best_hyper_img = imgs[best_hyper_index]
            cv2.imwrite(directory_path+"/full_hyperspec_img.png", best_hyper_img)
            return best_hyper_img, best_hyper_img.shape

        #check if all the hyperspectral image matrix have been read.
        try:
            hyperspectral_data = np.concatenate((hyperspectral_data, im[..., np.newaxis]), axis = 2)
        except:
            hyperspectral_data = im[..., np.newaxis]

        assert(sheet_cnt == hyperspectral_data.shape[2])
        print("________Read all the channels of hyperspectral image.__________")

        return imgs

#class for hyperspectral images preprocessing
# class HyperspecPreprocess:
def preprocess_hyperdata(data):
    """
        Preprocess the hyperspectral image matrix to align it according to the rgb image.
    """
    data = (data - np.min(data))*255/(np.max(data)-np.min(data))
    data = data.astype('uint8')
    resize_hs_img = cv2.resize(data, (1400, data.shape[0]))
    return resize_hs_img

#resize and rotate rgb image
def preprocess_rgb(rgb_img, hyp_img):
    #flip verticall
    final_flipped = cv2.flip(rgb_img, 1)
    #rotate counterclockwise = transpose + flip horizontally
    hor_flipped = cv2.flip(final_flipped, 0)
    transpose_im = cv2.transpose(hor_flipped)
    #resize rgb
    h, w = hyp_img.shape[:2]
    ratio = w/h
    # print(ratio)
    height = rgb_img.shape[0]
    width =  rgb_img.shape[1]
    # print(height)
    # print(width)
    rgb_img_resize = cv2.resize(transpose_im, (w, h))
    return rgb_img_resize


#Initiates class for image alignment utilizing the configuration file
class ImageAlignment:
    def __init__(self, grid_type):
        if grid_type == 1:
            self.h_matrix = cfg.h_matrix_1
        elif  grid_type == 2:
            self.h_matrix = cfg.h_matrix_2
        else:
            pass

    def warp_image(self, hyp_img, rgb_img, directory_path):
        # transform the images and overlay them to see if they align properly
        height, width = rgb_img.shape[:2]
        warped_rgb = cv2.warpPerspective(rgb_img, self.h_matrix, (width, height))

        align_img = cv2.addWeighted(warped_rgb, .3, hyp_img, .7, 1)

        unalign_img = cv2.addWeighted(rgb_img, .3, hyp_img, .7, 1)

        return align_img, unalign_img, warped_rgb, self.h_matrix


#This function reads the hyperspectral excel workbook
#and get the correct orientation of the hyper image
#and preprocess hyper and rgb image for alignment
def preprocess_hyper_and_rgb(hs_img, rgb_image, directory_path, sheet_number):
    #read hyperspectral data from excel workbook
    hyper_img = read_hyper_data(hs_img, directory_path,\
                                sheet_number, choose_best = False)

    #call functions to preprocess rgb image
    rgb_img = cv2.imread(rgb_image)
    rgb_prep = preprocess_rgb(rgb_img, hyper_img)
    cv2.imwrite(directory_path+"/rgb_prep.png", rgb_prep)
    print("------------Preprocess is saved and finished.-------------------")

    return rgb_prep, hyper_img


#This function calculates the homography matrix for the images:
def align_image(hyp_img, rgb_img):

    #preprocess rgb to hsv
    rgb_hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # Convert images to grayscale
    rgb_gray = cv2.cvtColor(rgb_hsv, cv2.COLOR_BGR2GRAY)
    hyp_gray = cv2.cvtColor(hyp_img, cv2.COLOR_BGR2GRAY)

    '''
    Tunable parameters:
    Modify the values in the cv2.adaptiveThreshold functions to obtain better h_matrix
    '''
    rgb_thresh = cv2.adaptiveThreshold(rgb_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,15,5)

    hyp_thresh = cv2.adaptiveThreshold(hyp_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    # cv2.imshow("rgbthresh", rgb_thresh)
    # cv2.waitKey(0)
    # cv2.imshow("hypthresh", hyp_thresh)
    # cv2.waitKey(0)

    '''
    Tunable parameters:
    Modify the values in the cv2.goodFeaturesToTrack function to obtain better h_matrix
    '''
    # find the coordinates of good features to track  in prep_rgb_img
    hyp_features = cv2.goodFeaturesToTrack(hyp_thresh, 10000, .1, 5)

    # find corresponding features in current photo
    rgb_features = np.array([])
    rgb_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(hyp_thresh, rgb_thresh, hyp_features, rgb_features, flags=1)

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

    align_img = cv2.addWeighted(warped_rgb, .3, hyp_img, .7, 1)

    unalign_img = cv2.addWeighted(rgb_img, .3, hyp_img, .7, 1)

    return align_img, unalign_img, warped_rgb, h_transformation

if __name__ == '__main__':
    print(cfg.h_matrix_2)
    pass
