'''
    Comments: cropping probably won't be required in this case. since this works with 
'''

import os
import sys
from PIL import Image
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

wd = sys.argv[1] #"/scratch2/NSF_GWAS/deeplab/deeplab/dataset/JPEG/EJ/"

class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]
idx_palette = np.reshape(np.asarray(class_color), (-1))


def cropshrinkIMG(file_path):
    try:
        if '.jpg' in file_path:
            file_in = Image.open(file_path).convert("RGB")
            file_path = file_path.replace('rgb', 'rgb_cropped')
        elif '.png' in file_path: # for labels -- probably won't be needed
            file_in = imread(file_path, pilmode='P')
            file_in = Image.fromarray(file_in)
            file_path = file_path.replace('segment', 'segment_cropped')
        else:
            raise Exception(file_path.split('.')[-1]+' file type not supported!')

        file_in = file_in.resize((900,900))
        file_in = file_in.rotate(270)
        file_in = file_in.crop((0, 150, 900, 750))
        
        if '.png' in file_path:
            file_in.putpalette(list(idx_palette))

        file_in.save(file_path)
    except: 
        print('Corrupt ' + str(file_path))

def croploop(wd):
    file_list = os.listdir(wd)
    os.chdir(wd)
    file_list = [x for x in file_list if '.png' in x] + [x for x in file_list if '.jpg' in x]
    for file in file_list:
        cropshrinkIMG(file)

def main(wd):
    croploop(wd)

if __name__ == "__main__":
    main(wd = wd)
