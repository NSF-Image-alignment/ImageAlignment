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

### To increase brightness of the hyperspectral image, threshold the image and add it to the original hyperspectral image. Use parameters image_thresh_high and image_thresh_low to threshold the image. Use argument 'distance' to find good matches (default=0.6). Use argument 'gaussian_sigma' to modify sigma of the SIFT algorithm.
```bash
python main.py --hyper-img hyper_image_path --rgb-img rgb_image_path --mode 1 --image_thresh_high 120 --image_thresh_low 50 --distance 0.7 --gaussian_sigma 1.6
```



### To run for a set of images.
1. Modify the .csv file 

    Format:
    header - hyper_img,rgb_images
    filename1 - hyper_img_path,rgb_img_path
    filename2
    filename3
    
    
    (Example file is present in the test_pipeline folder)

2. Run the script with the command - 
```bash
python main.py --img-csv csv_rgb_hyper_image_paths --mode 2
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


### Instructions on how to install SIFT and SURF in OpenCV (build from the source) -
Detailed installation instructions, can be found [here](https://medium.com/repro-repo/install-opencv-4-0-1-from-source-on-macos-with-anaconda-python-3-7-to-use-sift-and-surf-9d4287d6228b)
 
1. Change the directory
```bash
cd ~
```

2. Clone the OpenCV and OpenCV-contrib repository
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

3. Make a folder build inside opencv
```bash
cd opencv
mkdir build
cd build
```

4. Build install using CMAKE and MAKE (Please note the parameter information) -
```bash
export CONDA_HOME=~/anaconda3 
export CPLUS_INCLUDE_PATH=$CONDA_HOME/envs/cv/lib/python3.7
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D PYTHON3_LIBRARY=$CONDA_HOME/envs/[envname]/lib/libpython3.7m.dylib \
    -D PYTHON3_INCLUDE_DIR=$CONDA_HOME/envs/[envname]/include/python3.7m \
    -D PYTHON3_EXECUTABLE=$CONDA_HOME/envs/[envname]/bin/python \
    -D PYTHON3_PACKAGES_PATH=$CONDA_HOME/envs/[envname]/lib/python3.7/site-packages \
    -D OPENCV_ENABLE_NONFREE=ON ..
make -j4
```

5. Install the files
```bash
sudo make install
```
