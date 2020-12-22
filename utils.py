#!usr/bin/python
###############################################################################
__authors__ = "Jia Yi Li, Damanpreet Kaur"
__description__ = "Utility functions for Image Alignment"

__date__ = "06/01/2020"
__maintainer__ = "Damanpreet Kaur"
__version__ = 1.2 

'''
Using SIFT algorithm for feature matching
'''
###############################################################################
import os
from xlrd import open_workbook, XL_CELL_NUMBER
import numpy as np
import cv2
from config import config as cfg
import time
from skimage import measure
from PIL import Image
from numpy import genfromtxt
import matplotlib.pyplot as plt

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


def preprocess_hyperdata(data):
    """
        Preprocess the hyperspectral image matrix to align it according to the rgb image.
    """
    data = (data - np.min(data))*255/(np.max(data)-np.min(data))
    data = data.astype('uint8')
    resize_hs_img = cv2.resize(data, (1400, data.shape[0]))
    return resize_hs_img


#resize and rotate rgb image
def preprocess_rgb(rgb_img, h, w, ext):
    rgb_img = cv2.rotate(rgb_img, rotateCode=0)
    rgb_img = cv2.rotate(rgb_img, rotateCode=1)
    rgb_img = cv2.flip(rgb_img, 1)
    
    ratio = w/h
    # print(ratio)
    # height = rgb_img.shape[0]
    # width =  rgb_img.shape[1]
    if ext == 'jpg':
        rgb_img_resize = cv2.resize(rgb_img, (w, h))
    else:
        rgb_img_resize = Image.fromarray(rgb_img).resize((w, h), Image.NEAREST)
    
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
def preprocess_hyper_and_rgb(hyper_img, rgb_image, directory_path, sheet_number):
    #read hyperspectral data from a csv file
    # ext = hs_img.split('.')[-1]
    # if ext=='csv':
    #     hyper_img = genfromtxt(hs_img, delimiter=',')
    #     hyper_img = np.uint8(hyper_img)
    # elif ext=='jpg':
    #     hyper_img = cv2.imread(hs_img)
    # else:   #read hyperspectral data from excel workbook
    #     hyper_img = read_hyper_data(hs_img, directory_path,\
    #                             sheet_number, choose_best = False)

    #call functions to preprocess rgb image
    rgb_img = cv2.imread(rgb_image)
    if len(hyper_img.shape)==2: h, w = hyper_img.shape
    else: h, w, _ = hyper_img.shape
    rgb_prep = preprocess_rgb(rgb_img, h, w , 'jpg') 

    print("------------Preprocess is saved and finished.-------------------")

    return rgb_prep, hyper_img


#This function calculates the homography matrix for the images:
def align_image(hyp_img, rgb_img, distance=0.6, ch=-1, image_thresh_low=None, image_thresh_high=None, sigma=1.6):
    if ch==-1:
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)            # Convert images to grayscale
    else:
        rgb_gray = rgb_img[:,:,ch]                                      # select the channel from the rgb image

    # read hyperspectral image
    hyp_gray = hyp_img
    if len(hyp_img.shape)==3:
        hyp_gray = cv2.cvtColor(hyp_img, cv2.COLOR_BGR2GRAY)

    if image_thresh_low or image_thresh_high:
        _, hyp_thresh = cv2.threshold(hyp_gray, image_thresh_low, image_thresh_high, cv2.THRESH_BINARY)
        hyp_gray += hyp_thresh  

    '''
        Fine-tune params according to the image.
        n_features - Number of best features to retain. Will be 0.
        nOctaveLayers - Depends on the image size. 
                        How many octaves of the original image should be formed.
        Sigma - Gaussian kernel size. Modify according to the image resolution.
        edgeThreshold - Larger the edge threshold, more features are retained. 
                Depends on how strong the corners and edges are.
    '''
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, edgeThreshold=100, sigma=1.6) 
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, edgeThreshold=100, sigma=sigma) 
    
    # finding the keypoint descriptors
    kpts1, descs1 = sift.detectAndCompute(hyp_gray, None)
    kpts2, descs2 = sift.detectAndCompute(rgb_gray, None)
    
    # find matches
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 15), {}) 
    matches = matcher.knnMatch(descs1, descs2, 2) # find top2 matches for every descriptor
    matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches

    # retain the best matches
    good = [m1 for (m1, m2) in matches if m1.distance < distance * m2.distance]
    if len(good)<4:
        print("No of matches are less than 4. Cannot run RANSAC")
        exit()

    # transform the source points to match the destination points.
    dst_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    src_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # find homography matrix
    h_transformation, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=10)
    mask = mask.ravel().tolist()
    
    # uncomment the below section to test the keypoints. 
    # good = [g for i,g in enumerate(good) if mask[i]==1]
    # print("Length of good: ", len(good))
    # img = cv2.drawMatches(hyp_gray,kpts1,rgb_gray,kpts2,good,None)
    # plt.imshow(img)
    # plt.show()

    # apply the homography matrix 
    height, width = rgb_img.shape[:2]
    warpImage = cv2.warpPerspective(rgb_gray, h_transformation, (width, height))

    align_img = cv2.addWeighted(warpImage, .3, hyp_gray, .7, 1)
    unalign_img = cv2.addWeighted(rgb_gray, .3, hyp_gray, .7, 1)

    return align_img, unalign_img, warpImage, h_transformation

if __name__ == '__main__':
    my_data = genfromtxt('hs.csv', delimiter=',')
    cv2.imwrite("hyperspec_img.png", my_data)
