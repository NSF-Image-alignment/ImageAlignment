"""
    Utility functions for image matching task.
"""
import os
from xlrd import open_workbook  
from xlrd import XL_CELL_TEXT, XL_CELL_NUMBER, XL_CELL_DATE, XL_CELL_BOOLEAN
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        book = open_workbook(filename)
    except:
        print("Check if the file is present. If it present remove the quotes around the file name and try again.")
        exit()

    sheet_cnt = book.nsheets
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

    return data, sheet_cnt

class ChloroPreprocess:
    def __init__(self, sheet_chloro):
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 0, 0, 0
        self.sheet_chloro = sheet_chloro

    def crop_dimensions(self, data):
        th3 = cv2.adaptiveThreshold(data,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        # perform erosion to remove any outlier points
        kernel = np.ones((5,5),np.uint8)
        th3 = cv2.erode(th3,kernel,iterations = 1)
        # crop the portion of image
        self.min_x, self.max_x, self.min_y, self.max_y = min(np.where(th3==0)[0]), max(np.where(th3==0)[0]), min(np.where(th3==0)[1]), max(np.where(th3==0)[1]) 
        print("chlorophyll channel crop dimensions: ", self.min_x, self.max_x, self.min_y, self.max_y)

    def preprocess_chlorophyll_ch(self, data, sheet_no):
        """
            Preprocess the chlorophyll channel of the image to align it according to the rgb image.
        """
        data = (data - np.min(data))*255/(np.max(data)-np.min(data))
        data = data.astype('uint8')
        if sheet_no==self.sheet_chloro:
            self.crop_dimensions(data)
        cropped_im = data[self.min_x-5:self.max_x+5, self.min_y-5:self.max_y+5]
        transpose_im = cv2.transpose(cropped_im)
        flip_im = cv2.flip(cv2.flip(transpose_im, 0), 1)
        return flip_im

class RGBPreprocess:
    def __init__(self):
        self.th, self.tw, self.bh, self.bw = 0, 0, 0, 0

    def preprocess_greench_image(self, greench_img):
        """
            Preprocess the green channel of the rgb image to get the plants information.
            1. Crop the petri dish from the image.
            2. Median blur the image to remove the small noises in the image.
            3. Find contours to detect the plants in the petri dish.
            4. Crop the plants from the image.
        """
        # Resize image to detect the contours easily.
        self.h, self.w = 600, 600
        #h, w = int(greench_img.shape[0]/8), int(greench_img.shape[0]/8)
        greench_img = cv2.resize(greench_img, (self.h, self.w))

        circles = cv2.HoughCircles(greench_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=240, maxRadius=250)
        index = np.argmin(circles[circles[:,:,0] > 300][:,0])
        circle = circles[:,index,:]

        green = greench_img.copy()
        
        # draw the outer circle
        # cv2.circle(green,(circle[0,0],circle[0,1]),circle[0,2],(0,255,0),2)
        # # draw the center of the circle
        # cv2.circle(green,(circle[0,0],circle[0,1]),2,(0,0,255),3)
        # plt.imshow(green)
        # plt.show()

        # # Apply bilateral filter to remove the extra noise and smoothen the.
        # filtered_img = cv2.bilateralFilter(greench_img, 5, 175, 175)

        # # Canny detector to detect the canny edges and find contours.
        # edge_detected_img = cv2.Canny(filtered_img, 75, 200)
        # _, contours, _= cv2.findContours(edge_detected_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # # crop to the petri dish in the image.
        # for contour in contours:
        #     approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        #     area = cv2.contourArea(contour)
        #     if ((len(approx) > 8) & (area > 1000)):
        #         contour_list = contour
        #         break

        # # use the radius and center of the image.
        # (x, y), radius = cv2.minEnclosingCircle(contour_list)

        (x, y), radius = (circle[0, 0], circle[0, 1]), circle[0, 2]
        center = (int(x), int(y))
        radius = int(radius)
        print("RGB center: ", center)
        print("RGB radius: ", radius)
        greench_img = cv2.circle(greench_img, center, radius, (0,255,0),  2)

        # Change image to black and white to easily detect contours.
        # Modify the values of thresholding based on image pixel intensities. 
        img = cv2.medianBlur(greench_img, 5) # apply Median blur to smoothen out the noise (small light visible plants)
        ret, thresh = cv2.threshold(img, 100, 255, 0)

        # find contours to detect the plants inside the circle
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        

        contour_pts = []
        for contour in contours:
            for c in contour:               
                if ((c[0][0]-center[0])**2 + (c[0][1]-center[1])**2) < (radius-15)**2:          
                    contour_pts.append(c[0])

        contour_pts = np.array(contour_pts)
        self.th, self.tw, self.bh, self.bw = np.min(contour_pts[:, 0]), np.min(contour_pts[:, 1]), np.max(contour_pts[:, 0]), np.max(contour_pts[:, 1])
        print("RGB crop dimensions: ", self.th, self.tw, self.bh, self.bw)

        img = greench_img[self.tw:self.bw, self.th:self.bh]
        
        return img
    
    def preprocess_rgb(self, image):
        image = cv2.resize(image, (self.h, self.w))
        image = image[self.tw:self.bw, self.th:self.bh, :]
        return image

class ImageAlignment:
    def __init__(self):
        self.h_matrix = 0

    def image_align_sift(self, img1, img2):
        """
            Warp the chlorophyll channel image to align with the green channel of the rgb image.
            img1 - chlorophyll channel of the image.
            img2 - green channel of the rgb image.
        """
        
        # resize the images
        h, w = int(img1.shape[1]*1.2), int(img1.shape[0]*1.2)
        img2 = cv2.resize(img2, (h, w))
        img1 = cv2.resize(img1, (h, w))

        print("Shape of image1: ", img1.shape)
        print("Shape of image2: ", img2.shape)

        # Create SIFT object
        """
            Fine-tune params according to the image.
            n_features - Number of best features to retain. Will be 0.
            nOctaveLayers - Depends on the image size. 
                            How many octaves of the original image should be formed.
            Sigma - Gaussian kernel size. Modify according to the image resolution.
            edgeThreshold - Larger the edge threshold, more features are retained. 
                    Depends on how strong the corners and edges are.
        """
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=5, edgeThreshold=10, sigma=1.6) #for 1.2 times the image size
        
        # sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=5, edgeThreshold=10, sigma=2) # for double the image size
        # sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=6, edgeThreshold=10, sigma=1.6) # for original image size

        # Create flann matcher
        matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {}) #algoirthm =1 

        # Detect keypoints and compute keypoint descriptors
        kpts1, descs1 = sift.detectAndCompute(img1, None)
        kpts2, descs2 = sift.detectAndCompute(img2, None)

        # knnMatch to get the top matching descriptors
        matches = matcher.knnMatch(descs1, descs2, 2)

        # Sort by their distance.
        matches = sorted(matches, key = lambda x:x[0].distance)

        # Ratio test, to get good matches.
        # fine-tune the distance according to the image. 
        # Greater the value more matches will be obtained. Could contain false negatives too.
        good = [m1 for (m1, m2) in matches if m1.distance < 0.8 * m2.distance]
        print('Length of good: ', len(good))

        # if no good matches were found.
        if len(good) < 4:
            print("No good matches found.")
            return img2

        # find the keypoints locations
        src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        print("Source points: ", src_pts)
        print("Destination points: ", dst_pts)

        matched = cv2.drawMatches(img1,kpts1,img2,kpts2,good,None)
        cv2.imwrite("save/matched.png", matched)

        # find homegraphy matrix
        self.h_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        
        print("Homography matrix: ", self.h_matrix)
        print("Image shape: ", img2.shape)

        warpImage = cv2.warpPerspective(img1, self.h_matrix, (img2.shape[1], img2.shape[0]))
        plt.imshow(warpImage)
        plt.show()

    def warp_image(self, hyperspectral_data, rgb_image, sheet_cnt, directory_path):
        # apply the homography to the image to obtain the warped image.
        # Please note - we are warping the chlorophyll channel according to the green channel image.
        h, w = int(hyperspectral_data.shape[1]*1.2), int(hyperspectral_data.shape[0]*1.2)
        rgb_image = cv2.resize(rgb_image, (h, w))
        hyperspectral_data = cv2.resize(hyperspectral_data, (h, w))

        for i in range(sheet_cnt):
            # img1 = hyperspectral_data[:,:,i]
            img1 = hyperspectral_data
            warpImage = cv2.warpPerspective(img1, self.h_matrix, (rgb_image.shape[1], rgb_image.shape[0]))
            
            try:
                if warpChloroImage is not None:
                    warpChloroImage = np.concatenate((warpChloroImage, warpImage[..., np.newaxis]), axis=2)
            except:
                warpChloroImage = warpImage[..., np.newaxis]

        # for the purpose of verifying if the obtained warped image aligns properly.
        cv2.imwrite(directory_path+"/warp_image.png", warpImage)
        new_img = np.hstack((hyperspectral_data, rgb_image[:,:,1]))
        cv2.imwrite(directory_path+"/unaligned_image.png", new_img)
        new_img = np.hstack((warpChloroImage[:,:,1], rgb_image[:,:,1]))
        cv2.imwrite(directory_path+"/hstack.png", new_img)
        new_img = np.vstack((warpChloroImage[:,:,0], rgb_image[:,:,1]))
        cv2.imwrite(directory_path+"/vstack.png", new_img)

        # stack the aligned images together
        # stackImg = np.concatenate((warpImage[..., np.newaxis], img2[..., np.newaxis], np.zeros((w, h, 1))), axis=2)
        stackImg = np.concatenate((warpChloroImage, rgb_image), axis=2)
        return stackImg
    