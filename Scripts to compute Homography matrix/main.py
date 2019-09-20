import cv2
import argparse
import numpy as np
from utils import sheet_to_array 
from skimage.transform import match_histograms
from utils import ChloroPreprocess, RGBPreprocess, ImageAlignment
import pickle
import os

CHLORO_CH = './Alignment_testing/GWO1_I2.0_F1.9_L80_103704_3_0_4.xlsx'
RGB_IMG = './Alignment_testing/GWO1_I2.0_F1.9_L80_103704_3_0_4_rgb.jpg'

SHEET_NO = 1

def get_arguments():
    """
        Parse all the command line arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--rgb-img", type=str, default=RGB_IMG, help="RGB image")
    parser.add_argument("--chloro-ch", type=str, default=CHLORO_CH, help="Chlorophyll matrix")
    parser.add_argument("--sheet-no", type=int, default=SHEET_NO, help="Chlorophyll channel sheet no according to which the images should be aligned.")
    return parser.parse_args()

def main():
    print("Arguments: ")
    args = get_arguments()
    print("Reading Chlorophyll matrix from sheet %d from %s"%(args.sheet_no, args.chloro_ch))
    print("RGB image: ", args.rgb_img)

    DIR_NAME = '.'.join(args.rgb_img.split('/')[-1].split('.')[:-1])
    directory_path = os.path.join('save', DIR_NAME)
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    # Read the chlorophyll image
    # Read sheet 1 chlorophyll channel
    ch = ChloroPreprocess(args.sheet_no)
    data, sheet_cnt = sheet_to_array(args.chloro_ch, args.sheet_no)
    print("Read chlorophyll data.")
    cv2.imwrite(directory_path+"/chloro"+str(args.sheet_no)+".png", data)

    chloro_im = ch.preprocess_chlorophyll_ch(data, args.sheet_no)
    cv2.imwrite(directory_path+"/cropped_chloro"+str(args.sheet_no)+".png", chloro_im)
    spectral_data = chloro_im[..., np.newaxis]
    hyperspectral_data = spectral_data

    for i in range(sheet_cnt):
        if i!=args.sheet_no:
            data, _ = sheet_to_array(args.chloro_ch, i)
            im = ch.preprocess_chlorophyll_ch(data, i)
            cv2.imwrite(directory_path+"/cropped_chloro"+str(i)+".png", im)

            try:
                hyperspectral_data = np.concatenate((hyperspectral_data, im[..., np.newaxis]), axis = 2)
            except:
                print("error!")
                hyperspectral_data = im[..., np.newaxis]

        else:
            try:
                hyperspectral_data = np.concatenate((hyperspectral_data, spectral_data), axis = 2)
            except:
                hyperspectral_data = spectral_data

    print(hyperspectral_data.shape)
    print("Read all the channels of hyperspectral image.")

    # Read the green channel from the rgb image and preprocess it.
    rgb = RGBPreprocess()
    rgb_img = cv2.imread(args.rgb_img)
    greench_img = rgb_img[:,:,1]
    greench_im = rgb.preprocess_greench_image(greench_img)
    cv2.imwrite(directory_path+"/green_ch.png", greench_im)

    rgb_cropim = rgb.preprocess_rgb(rgb_img)
    cv2.imwrite(directory_path+"/rgb_cropped.png", rgb_cropim)

    # Match the histograms of the source and reference image
    ch_im = cv2.resize(spectral_data, (greench_im.shape[1], greench_im.shape[0]))
    greench_eq = match_histograms(greench_im, ch_im, multichannel=False)
    greench_eq = np.round(greench_eq).astype(np.uint8)
    cv2.imwrite(directory_path+"/hist_match_green_ch.png", greench_eq)

    # find matches using SIFT and warp the chlorophyll channel to align with the green channel of the rgb image.  
    align = ImageAlignment()
    align.image_align_sift(spectral_data, greench_eq.astype(np.uint8))
    
    aligned_im = align.warp_image(hyperspectral_data, rgb_cropim, sheet_cnt, directory_path)

    with open(directory_path+"/arr_dump.pickle", "wb") as f:
        pickle.dump(aligned_im, f)

if __name__ == "__main__":
    main()
