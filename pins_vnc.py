import xml.etree.ElementTree as ET

import os, sys
import numpy as np
import random, zipfile, requests

from sklearn.model_selection import train_test_split


import torch
import cv2

# training a few blank tile, testing remove all
if getattr(sys, 'frozen', False):
    # The application is running as a bundled executable
    current_dir = os.path.dirname(sys.executable)
else:
    # The application is running as a script
    current_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.isdir(os.path.join(current_dir, "openslide-win64-20230414")):
    url = "https://github.com/openslide/openslide-winbuild/releases/download/v20230414/openslide-win64-20230414.zip"
    filename = "openslide-win64-20230414.zip"

    # Send a GET request to the URL and stream the response
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        with open(os.path.join(current_dir, filename), "wb") as file:
            # Iterate over the response content in chunks and write to file
            for chunk in response.iter_content(chunk_size=4096):
                file.write(chunk)
        print(f"Download completed: {filename}")
    else:
        print("Failed to download the file.")
    with zipfile.ZipFile(os.path.join(current_dir, 'openslide-win64-20230414.zip'), 'r') as zip:
        zip.extractall(current_dir)

from verification_dump import get_referance, slideRead
OPENSLIDE_PATH = os.path.join(current_dir, 'openslide-win64-20230414', 'bin')
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

blue_list = []
red_plus_list = []
red_minus_list = []


NM_P = 221      # number of pixels in a nanometer
TILE_SIZE = 1024
MICRO_TILE_SIZE = 226
RED_PER_CASE = 30
BLUE_PER_CASE = 30

for WSI_PATH in os.listdir('slides'):
    WSI_PATH = os.path.join('slides', WSI_PATH)
    if WSI_PATH.endswith('.ndpi'):
        XML_PATH = WSI_PATH + ".ndpa"
        character = '+'

        try:
            file = open(XML_PATH, "r")
            file_content = file.read()
            file.close()

            if character in file_content:
                plus = True
            else:
                plus = False
        except FileNotFoundError:
            print("File not found.")
        except IOError:
            print("Error reading the file.")

        XML_PATH = WSI_PATH + '.ndpa'
        slide = slideRead(WSI_PATH)
        LEVEL = slide.get_best_level_for_downsample(1.0 / 40)
        X_Reference, Y_Reference = get_referance(WSI_PATH, NM_P)

        # Load the XML file
        tree = ET.parse(XML_PATH)
        root = tree.getroot()

        # Find all 'annotation' elements with type='pin'
        red_pins = root.findall(".//annotation[@color='#ff0000']")
        blue_pins = root.findall(".//annotation[@color='#0000ff']")

        # Extract the pin coordinates
        red_pin_locations = []
        for pin in red_pins:
            x = int((int(pin.find('x').text) + X_Reference)/NM_P)
            y = int((int(pin.find('y').text) + Y_Reference)/NM_P)
            red_pin_locations.append((x, y))
        if len(red_pin_locations) > RED_PER_CASE:
            red_pin_locations = random.sample(red_pin_locations, RED_PER_CASE)

        blue_pin_locations = []
        for pin in blue_pins:
            x = int((int(pin.find('x').text) + X_Reference)/NM_P)
            y = int((int(pin.find('y').text) + Y_Reference)/NM_P)
            blue_pin_locations.append((x, y))
        if len(blue_pin_locations) > BLUE_PER_CASE:
            blue_pin_locations = random.sample(blue_pin_locations, BLUE_PER_CASE)

        # Print the extracted pin locations
        for location in red_pin_locations:
            cx = int(location[0] * (MICRO_TILE_SIZE / TILE_SIZE))
            cy = int(location[1] * (MICRO_TILE_SIZE / TILE_SIZE))
            im_roi = slide.read_region((((cx - cx % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE, ((cy - cy % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
            file = im_roi.convert('RGB')
            np_img = np.array(file)
            if plus:
                red_plus_list.append(np_img)
            else:
                red_minus_list.append(np_img)

        for location in blue_pin_locations:
            # cx = int(location[0])
            # cy = int(location[1])
            # im_roi = slide.read_region((cx - cx % TILE_SIZE, cy - cy % TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
            cx = int(location[0] * (MICRO_TILE_SIZE / TILE_SIZE))
            cy = int(location[1] * (MICRO_TILE_SIZE / TILE_SIZE))
            im_roi = slide.read_region((((cx - cx % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE, ((cy - cy % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
            file = im_roi.convert('RGB')
            np_img = np.array(file)
            blue_list.append(np_img)


print(len(blue_list))
print(len(red_plus_list))
print(len(red_minus_list))
# blue_list and red_list contain all the images

# Combine the lists into a single dataset
X = np.concatenate((blue_list, red_plus_list, red_minus_list))
y = np.concatenate((np.zeros(len(blue_list)), np.ones(len(red_plus_list)), np.ones(len(red_minus_list)) * 2))

# Split the dataset into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Print the sizes of each set
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

# Convert the labels to integers
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# dump all blue training data as images into ./tiles/train/blue
# dump all redplus training data as images into ./tiles/train/redplus
# dump all redminus training data as images into ./tiles/train/redminus

# similarly, dump testing and validation data into ./tiles/test and ./tiles/val

# Define the output directories
output_dir = os.path.join(current_dir, 'tiles')
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
val_dir = os.path.join(output_dir, 'val')

# Create the output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Dump training data
for i, image in enumerate(X_train):
    label = y_train[i]
    if label == 0:
        class_dir = os.path.join(train_dir, 'blue')
    elif label == 1:
        class_dir = os.path.join(train_dir, 'redplus')
    else:
        class_dir = os.path.join(train_dir, 'redminus')
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f'image_{i}.jpg')
    # Save image to the appropriate class directory
    # Assuming image is a PIL image
    # image.save(image_path)
    cv2.imwrite(image_path, image)

# Dump validation data
for i, image in enumerate(X_val):
    label = y_val[i]
    if label == 0:
        class_dir = os.path.join(val_dir, 'blue')
    elif label == 1:
        class_dir = os.path.join(val_dir, 'redplus')
    else:
        class_dir = os.path.join(val_dir, 'redminus')
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f'image_{i}.jpg')
    # Save image to the appropriate class directory
    # Assuming image is a PIL image
    # image.save(image_path)
    cv2.imwrite(image_path, image)

# Dump testing data
for i, image in enumerate(X_test):
    label = y_test[i]
    if label == 0:
        class_dir = os.path.join(test_dir, 'blue')
    elif label == 1:
        class_dir = os.path.join(test_dir, 'redplus')
    else:
        class_dir = os.path.join(test_dir, 'redminus')
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f'image_{i}.jpg')
    # Save image to the appropriate class directory
    # Assuming image is a PIL image
    # image.save(image_path)
    cv2.imwrite(image_path, image)

#!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/

if(torch.cuda.is_available()):
    print("CUDA is present")
else:
    print("CUDA is absent")

print(torch.cuda.current_device())
torch.cuda.get_device_name(0)

#%pip install -r requirements.txt

# Train YOLOv5s Classification on Imagenette160 for 3 epochs
#!python classify/train.py --model yolov5s-cls.pt --data ./tiles/train --epochs 5 --img 224 --cache
#!python classify/predict.py --weights runs/train-cls/exp2/weights/best.pt --img 224 --source ./tiles/test  --name exp_images