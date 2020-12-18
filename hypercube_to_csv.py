from spectral import *
from PIL import Image
import numpy as np

def main(hdr_path, cap, index, outpath_csv, outpath_img):
    print("Reading hyperspectral image " + str(hdr_path))
    image1 = envi.open(hdr_path)

    print("Extracting channel at index " + str(index))
    
    CLS_matrix = np.squeeze(image1[:,:,index],
                            axis=2)

    CLS_matrix = np.interp(CLS_matrix,
                           (0,
                            cap),
                           (0,
                            255)).astype(int)
    
    print("Writing channel matrix to " + str(outpath_csv))
    
    np.savetxt(outpath_csv,
               CLS_matrix,
               delimiter = ",")
    
    print("Making image for inspection and writing to " + str(outpath_img))
    
    CLS_matrix_expanded = np.expand_dims(CLS_matrix, axis=2)
    CLS_matrix_expanded_filled = np.concatenate((CLS_matrix_expanded,
                                                 CLS_matrix_expanded,
                                                 CLS_matrix_expanded),
                                                axis=2).astype(np.uint8)

    img = Image.fromarray(CLS_matrix_expanded_filled, 'RGB')
    
    img.save(fp = outpath_img,
             format = "JPEG")
    
if __name__== "__main__":
    main(hdr_path = sys.argv[1],
         cap = int(sys.argv[2]),
         index = int(sys.argv[3]),
         outpath_csv = sys.argv[4],
         outpath_img = sys.argv[5])