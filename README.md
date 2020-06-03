# ImageAlignment
Align the hyperspectral image with the RGB image

### To install the dependencies using anaconda, use the .yaml file. 
To create an environment and install the dependencies using anaconda, run the command -
```bash
conda env create -f align.yaml
```

### Create an output folder in the image_alignment folder using the command -
```bash
mkdir output
```

### To run the scripts for a single RGB image and hyperspectral matrix pair, use the command (to compute homography matrix):
```bash
python main.py --hyper-img hyper_image_path --rgb-img rgb_image_path --mode 1
```

### To run for a set of images.
1. Modify the .csv file 

    Format:
    header - rgb_images
    filename1 - rgb_img_path
    filename2
    filename3
    
    
    (Example file is present in the test_pipeline folder)

2. Run the script with the command - 
```bash
python main.py --hyper-img hyperspectral_image_path --img-csv csv_rgb_image_paths --mode 2
```
    
  
### The directory structure before running the scripts should look like -
ImageAlignment

+-- config.py

+-- Alignment_testing (Folder where RGB image and hyperspectral excel file is stored.)

|   +-- RGB_image file

|   +-- Hyperspectral matrix excel file

+-- Output (Script will automatically create subdirectories for each of the example images.)

+-- utils.py

+-- test.csv

+-- README.md

+-- main.py
        
