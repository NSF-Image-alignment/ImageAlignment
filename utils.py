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
import pandas as pd
import time
from skimage import measure

def sheet_to_array(filename):
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

    sheet_cnt = book.nsheets

    datas = []
    for sheet_number in range(sheet_cnt):
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

        datas.append(data)

    # print(len(datas))
    return datas, sheet_cnt

#function to read data from .xslx hyperspectral files
def read_hyper_data(hs_img):
    #initialize a HyperspecPreprocess varibale using the grid_type
    hyp_prep = HyperspecPreprocess()
    print("-------Begining to read hyperspectral data----------- ")
    start_time = time.time()
    datas, sheet_cnt = sheet_to_array(hs_img)
    end_time = time.time()
    print("total time taken during reading count_worksheets " + str(end_time - start_time))
    imgs = []
    #iterates through the number of channels
    for i in range(len(datas)):
        #reads data value from the sheet of hs_img
        data = datas[i]
        #preprocess the hs_img using data read from sheet
        im = hyp_prep.preprocess_hyperspec_ch(data)
        imgs.append(im)
        #calculate the entropy of image
        entropy = measure.shannon_entropy(im)
        # print("entropy"+str(i), entropy)
        entropies.append(entropy)
        #output preprocessed hs_img to path
        # cv2.imwrite(directory_path+"/hyperspec_img_"+str(i)+".png", im)

    #output best preprocessed hs_img to path
    best_hyper_index = np.argmax(entropies)
    best_hyper_img = imgs[best_hyper_index]
    cv2.imwrite(directory_path+"/hyperspec_img.png", best_hyper_img)

    try:
        hyperspectral_data = np.concatenate((hyperspectral_data, im[..., np.newaxis]), axis = 2)
    except:
        hyperspectral_data = im[..., np.newaxis]

    #check if all the hyperspectral image matrix have been read.
    assert(sheet_cnt == hyperspectral_data.shape[2])
    print("Read all the channels of hyperspectral image.")

    return best_hyper_img

#class for hyperspectral images preprocessing
class HyperspecPreprocess:
    def preprocess_hyperspec_ch(self, data):
        """
            Preprocess the hyperspectral image matrix to align it according to the rgb image.
        """
        data = (data - np.min(data))*255/(np.max(data)-np.min(data))
        data = data.astype('uint8')
        resize_hs_img = cv2.resize(data, (600,600))
        #rotate counterclockwise = transpose + flip horizontally
        transpose_im = cv2.transpose(resize_hs_img)
        hor_flipped = cv2.flip(transpose_im, 0)
        #flip vertically
        final_flipped = cv2.flip(hor_flipped, 1)

        return final_flipped

class RGBPreprocess:
    # def __init__(self, grid_type):
    #     if grid_type == 1:
    #         (self.th, self.tw, self.bh, self.bw) = cfg.rgb_cropdims_1
    #     else:
    #         (self.th, self.tw, self.bh, self.bw) = cfg.rgb_cropdims_2
    #     self.h, self.w = 600, 600
    #     print("RGB crop dimensions: ", self.th, self.tw, self.bh, self.bw)

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

    #resize rgb image
    def preprocess_rgb(self, image):
        image = cv2.resize(image, (600, 600))
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
        # h, w = int(hyperspectral_data.shape[1]*1.2), int(hyperspectral_data.shape[0]*1.2)
        # re_size = (600,600)
        # rgb_image = cv2.resize(rgb_image, re_size)
        # hyperspectral_data = cv2.resize(hyperspectral_data,re_size)

        # cv2.imshow("hyperspectral_data in warp", hyperspectral_data)
        # cv2.waitKey(0)

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
