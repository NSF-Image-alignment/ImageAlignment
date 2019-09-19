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
(this commands works for the unix and MAC OS system)


### To run the scripts for a single RGB image and hyperspectral matrix pair, use the command:
```bash
python main.py --rgb-img rgb_image_name_with_path --hyper-img hyperspectral_image_name_with_path
```
Example:
```bash
python main.py --rgb-img './Alignment_testing/GWO1_I2.0_F1.9_L80_103450_0_0_0_rgb.jpg' --hyper-img './Alignment_testing/GWO1_I2.0_F1.9_L80_103450_0_0_0.xlsx'
```

You can optionally modify these parameters in the script. Modify the HYPERSPECTRAL_IMG and RGB_IMG parameter at the top of the main.py.

Run the script as -
```bash
python main.py
```

### To run for a set of images.
1. Modify the test.csv file 

    Format:
    
    hyperspectral_img_with_path rgb_img_with_path
    
    (Example file is present in the folder)

2. Run the script with the command - 
    ```bash
    python main.py --csv --csv-name 'test.csv'
    ```
    
    You can optionally mention the csv name in the main.py script. Modify the parameter CSV_NAME at the top of the script.

    Run the script as -
    ```bash
    python main.py --csv
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
        
