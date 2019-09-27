import cv2
import argparse
import numpy as np
from utils import sheet_to_array, count_worksheets
# from skimage.transform import match_histograms
from utils import HyperspecPreprocess, RGBPreprocess, ImageAlignment
import pickle
import os

HYPERSPECTRAL_IMG = './Alignment_testing/CV2_F1.9_I5.0_L100_cyan_234342_19_2_5-CLS1.out.xlsx'
RGB_IMG = './Alignment_testing/CV2_F1.9_I5.0_L100_cyan_234342_19_2_5_rgb.png'
CSV_NAME = './test.csv'
GRID_TYPE = 2 

def get_arguments():
    """
        Parse all the command line arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--rgb-img", type=str, default=RGB_IMG, help="RGB image")
    parser.add_argument("--hyper-img", type=str, default=HYPERSPECTRAL_IMG, help="Chlorophyll matrix")
    parser.add_argument("--grid-type", type=int, default=GRID_TYPE, help="Grid type options. 1 for a 3X3 grid and 2 for other grid type.")
    parser.add_argument("--csv", action="store_true", help="Passing file names in CSV.")
    parser.add_argument("--csv-name", type=str, default=CSV_NAME, help="CSV file name.")
    return parser.parse_args()

def align_images(hs_img, rgb_image, grid_type):
    # create the output directory
    DIR_NAME = '.'.join(rgb_image.split('/')[-1].split('.')[:-1])
    directory_path = os.path.join('output', DIR_NAME)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    # Read the chlorophyll image 
    sheet_cnt = count_worksheets(hs_img)

    ch = HyperspecPreprocess(grid_type)

    for i in range(sheet_cnt):
        data = sheet_to_array(hs_img, i)
        im = ch.preprocess_hyperspec_ch(data)
        cv2.imwrite(directory_path+"/cropped_hyperspec_img_"+str(i)+".png", im)

        try:
            hyperspectral_data = np.concatenate((hyperspectral_data, im[..., np.newaxis]), axis = 2)
        except:
            hyperspectral_data = im[..., np.newaxis]

    print(hyperspectral_data.shape)
    
    assert(sheet_cnt == hyperspectral_data.shape[2]) #check if all the hyperspectral image matrix have been read.
    print("Read all the channels of hyperspectral image.")

    # Read the green channel from the rgb image and preprocess it.
    rgb = RGBPreprocess(grid_type)
    rgb_img = cv2.imread(rgb_image)
    rgb_cropim = rgb.preprocess_rgb(rgb_img)
    cv2.imwrite(directory_path+"/rgb_cropped.png", rgb_cropim)

    # find matches using SIFT and warp the chlorophyll channel to align with the green channel of the rgb image.  
    align = ImageAlignment(grid_type)    
    aligned_im = align.warp_image(hyperspectral_data, rgb_cropim, sheet_cnt, directory_path)

    with open(directory_path+"/arr_dump.pickle", "wb") as f:
        pickle.dump(aligned_im, f)

def main():
    args = get_arguments()

    if args.csv:
        with open(args.csv_name, 'rt') as f:
            files = f.readlines()
            for fs in files:
                hs_img, rgb_img, grid_type = fs.split(',')
                print("Arguments: ")
                print("Reading hyperspectral image matrix from %s"%(hs_img))
                print("RGB image: ", rgb_img)
                hs_img = hs_img.replace("'", '')
                rgb_img = rgb_img.replace("'", '')
                align_images(hs_img, rgb_img.rstrip(), grid_type.rstrip())
    else:
        hs_img = args.hyper_img
        rgb_img = args.rgb_img
        grid_type = args.grid_type
        hs_img = hs_img.replace("'", '')
        rgb_img = rgb_img.replace("'", '')

        print("Arguments: ")
        print("Reading hyperspectral image matrix from %s"%(hs_img))
        print("RGB image: ", rgb_img)

        align_images(hs_img, rgb_img, grid_type)
    

if __name__ == "__main__":
    main()
