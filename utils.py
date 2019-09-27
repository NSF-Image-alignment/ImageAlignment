"""
    Utility functions for image matching task.
"""
import os
from xlrd import open_workbook  
from xlrd import XL_CELL_TEXT, XL_CELL_NUMBER, XL_CELL_DATE, XL_CELL_BOOLEAN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from config import config as cfg

def count_worksheets(filename):
    # sheet
    try:
        book = open_workbook(filename, on_demand=True)
    except:
        print("Check if the file is present. If it present remove the quotes around the file name and try again.")
        exit()

    sheet_cnt = book.nsheets
    return sheet_cnt

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
    
    # sheet
    try:
        book = open_workbook(filename, on_demand=True)
    except:
        print("Check if the file is present. If it present remove the quotes around the file name and try again.")
        exit()

    sheet = book.sheet_by_index(sheet_number)
    rows = sheet.nrows

    # cols
    last_col = sheet.ncols
    data = np.empty([last_col, rows], dtype=int)

    cols = [col for col in range(0, last_col + 1)]

    for row in range(0, sheet.nrows):
        row_values = sheet.row(row)
        for col, cell in enumerate(row_values):            
            if col in cols and cell.ctype == XL_CELL_NUMBER:
                data[col, row] = cell.value              

    return data

class HyperspecPreprocess:
    def __init__(self, grid_type):
        if grid_type == 1:
            (self.min_x, self.max_x, self.min_y, self.max_y) = cfg.hyperspec_cropdims_1
        else:
            (self.min_x, self.max_x, self.min_y, self.max_y) = cfg.hyperspec_cropdims_2
        print("Hyperspectral image matrix crop dimensions: ", self.min_x, self.max_x, self.min_y, self.max_y)

    def preprocess_hyperspec_ch(self, data):
        """
            Preprocess the hyperspectral image matrix to align it according to the rgb image.
        """
        data = (data - np.min(data))*255/(np.max(data)-np.min(data))
        data = data.astype('uint8')
        cropped_im = data[self.min_x-5:self.max_x+5, self.min_y-5:self.max_y+5]
        transpose_im = cv2.transpose(cropped_im)
        flip_im = cv2.flip(cv2.flip(transpose_im, 0), 1)
        return flip_im

class RGBPreprocess:
    def __init__(self, grid_type):
        if grid_type == 1:
            (self.th, self.tw, self.bh, self.bw) = cfg.rgb_cropdims_1
        else:
            (self.th, self.tw, self.bh, self.bw) = cfg.rgb_cropdims_2
        self.h, self.w = 600, 600
        print("RGB crop dimensions: ", self.th, self.tw, self.bh, self.bw)

    def preprocess_greench_image(self, greench_img):
        """
            Preprocess the green channel of the rgb image to get the plants information.
            1. Crop the petri dish from the image.
            2. Median blur the image to remove the small noises in the image.
            3. Find contours to detect the plants in the petri dish.
            4. Crop the plants from the image.
        """
        # Resize image to detect the contours easily.
        greench_img = cv2.resize(greench_img, (self.h, self.w))
        circles = cv2.HoughCircles(greench_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=240, maxRadius=250)
        index = np.argmin(circles[circles[:,:,0] > 300][:,0])
        circle = circles[:,index,:]

        green = greench_img.copy()

        img = greench_img[self.tw:self.bw, self.th:self.bh]
        return img
    
    def preprocess_rgb(self, image):
        image = cv2.resize(image, (self.h, self.w))
        image = image[self.tw:self.bw, self.th:self.bh, :]
        return image

class ImageAlignment:
    def __init__(self, grid_type):
        if grid_type == 1:
            self.h_matrix = cfg.h_matrix_1
        else:
            self.h_matrix = cfg.h_matrix_2    
        print("Homography matrix: ", self.h_matrix)

    # def image_align_sift(self, img1, img2):
    #     """
    #         Warp the chlorophyll channel image to align with the green channel of the rgb image.
    #         img1 - chlorophyll channel of the image.
    #         img2 - green channel of the rgb image.
    #     """
        
    #     # resize the images
    #     h, w = int(img1.shape[1]*1.2), int(img1.shape[0]*1.2)
    #     img2 = cv2.resize(img2, (h, w))
    #     img1 = cv2.resize(img1, (h, w))

    #     print("Shape of image1: ", img1.shape)
    #     print("Shape of image2: ", img2.shape)
    #     print("Homography matrix: ", self.h_matrix)

    #     warpImage = cv2.warpPerspective(img1, self.h_matrix, (img2.shape[1], img2.shape[0]))       

    def warp_image(self, hyperspectral_data, rgb_image, sheet_cnt, directory_path):
        # apply the homography to the image to obtain the warped image.
        # Please note - we are warping the chlorophyll channel according to the green channel image.
        h, w = int(hyperspectral_data.shape[1]*1.2), int(hyperspectral_data.shape[0]*1.2)
        rgb_image = cv2.resize(rgb_image, (h, w))
        hyperspectral_data = cv2.resize(hyperspectral_data, (h, w))

        for i in range(sheet_cnt):
            img1 = hyperspectral_data[:,:,i]
            warpImage = cv2.warpPerspective(img1, self.h_matrix, (rgb_image.shape[1], rgb_image.shape[0]))
            
            try:                
                warpHyperImage = np.concatenate((warpHyperImage, warpImage[..., np.newaxis]), axis=2)
            except:
                warpHyperImage = warpImage[..., np.newaxis]

        # for the purpose of visualization if the obtained warped image aligns properly.
        new_img = np.hstack((warpHyperImage[:,:,3], rgb_image[:,:,1]))
        cv2.imwrite(directory_path+"/hstack.png", new_img)
        new_img = np.vstack((warpHyperImage[:,:,3], rgb_image[:,:,1]))
        cv2.imwrite(directory_path+"/vstack.png", new_img)
        

        # stack the aligned images together
        stackImg = np.concatenate((warpHyperImage, rgb_image), axis=2)
        return stackImg
    