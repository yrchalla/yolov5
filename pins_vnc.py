import xml.etree.ElementTree as ET

import os, sys
import numpy as np
import random, zipfile, requests
from PIL import Image

from sklearn.model_selection import train_test_split


import torch
import cv2
import torchvision.transforms as transforms

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

# X = np.array([])
# Y = np.array([])
X = []
Y = []


NM_P = 221      # number of pixels in a nanometer
TILE_SIZE = 1024
MICRO_TILE_SIZE = 226
RED_PER_CASE = 90
normal_PER_CASE = 45

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
        normal_pins = root.findall(".//annotation[@color='#0000ff']")

        # Extract the pin coordinates
        red_pin_locations = []
        for pin in red_pins:
            x = int((int(pin.find('x').text) + X_Reference)/NM_P)
            y = int((int(pin.find('y').text) + Y_Reference)/NM_P)
            red_pin_locations.append((x, y))
        if len(red_pin_locations) > RED_PER_CASE:
            red_pin_locations = random.sample(red_pin_locations, RED_PER_CASE)

        normal_pin_locations = []
        for pin in normal_pins:
            x = int((int(pin.find('x').text) + X_Reference)/NM_P)
            y = int((int(pin.find('y').text) + Y_Reference)/NM_P)
            normal_pin_locations.append((x, y))
        if len(normal_pin_locations) > normal_PER_CASE:
            normal_pin_locations = random.sample(normal_pin_locations, normal_PER_CASE)

        # Print the extracted pin locations
        for location in red_pin_locations:
            cx = int(location[0] * (MICRO_TILE_SIZE / TILE_SIZE))
            cy = int(location[1] * (MICRO_TILE_SIZE / TILE_SIZE))
            im_roi = slide.read_region((((cx - cx % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE, ((cy - cy % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
            file = im_roi.convert('RGB')
            np_img = np.array(file)
            if plus:
                # X = np.append(X, np_img)
                # Y = np.append(Y, 1)
                X.append(np_img)
                Y.append(1)
            else:
                # X = np.append(X, np_img)
                # Y = np.append(Y, 2)
                X.append(np_img)
                Y.append(2)

        for location in normal_pin_locations:
            # cx = int(location[0])
            # cy = int(location[1])
            # im_roi = slide.read_region((cx - cx % TILE_SIZE, cy - cy % TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
            cx = int(location[0] * (MICRO_TILE_SIZE / TILE_SIZE))
            cy = int(location[1] * (MICRO_TILE_SIZE / TILE_SIZE))
            im_roi = slide.read_region((((cx - cx % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE, ((cy - cy % MICRO_TILE_SIZE) * TILE_SIZE) // MICRO_TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
            file = im_roi.convert('RGB')
            np_img = np.array(file)
            # X = np.append(X, np_img)
            # Y = np.append(Y, 0)
            X.append(np_img)
            Y.append(0)

# normal_list and red_list contain all the images
normalCount = Y.count(0)
plusCount = Y.count(1)
minusCount = Y.count(2)

if minusCount > min(plusCount, minusCount, normalCount):
    p = min(plusCount, minusCount, normalCount) / minusCount
    for i in range(len(Y) - 1, -1, -1):
        if (Y[i] == 2) and random.random() > p:
            Y.pop(i)
            X.pop(i)
if plusCount > min(plusCount, minusCount, normalCount):
    p = min(plusCount, minusCount, normalCount) / plusCount
    for i in range(len(Y) - 1, -1, -1):
        if (Y[i] == 1) and random.random() > p:
            Y.pop(i)
            X.pop(i)
if normalCount > min(plusCount, minusCount, normalCount):
    p = min(plusCount, minusCount, normalCount) / normalCount
    for i in range(len(Y) - 1, -1, -1):
        if (Y[i] == 0) and random.random() > p:
            Y.pop(i)
            X.pop(i)

print("Normal tiles: ", Y.count(0))
print("Plus tiles: ", Y.count(1))
print("Minus tiles: ", Y.count(2))


# Combine the lists into a single dataset
# X = np.array(X)
# Y = np.array(Y)



# Split the dataset into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the data augmentation transforms
# augmentation_transforms = transforms.Compose([
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(45),
# ])

# # Apply data augmentation to each image
# augmented_X_train_val = []
# for image in X_train_val:
#     image = Image.fromarray(image)
#     augmented_image = augmentation_transforms(image)
#     augmented_X_train_val.append(np.array(augmented_image))

# # Extend the training set with augmented images
# X_train_val.extend(augmented_X_train_val)
# y_train_val.extend(y_train_val)


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Print the sizes of each set
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Testing set size:", len(X_test))

# Convert the labels to integers
# y_train = y_train.astype(int)
# y_val = y_val.astype(int)
# y_test = y_test.astype(int)

# dump all normal training data as images into ./tiles/train/normal
# dump all MSIplus training data as images into ./tiles/train/MSIplus
# dump all MSIminus training data as images into ./tiles/train/MSIminus

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
        class_dir = os.path.join(train_dir, 'normal')
    elif label == 1:
        class_dir = os.path.join(train_dir, 'MSIplus')
    else:
        class_dir = os.path.join(train_dir, 'MSIminus')
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
        class_dir = os.path.join(val_dir, 'normal')
    elif label == 1:
        class_dir = os.path.join(val_dir, 'MSIplus')
    else:
        class_dir = os.path.join(val_dir, 'MSIminus')
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
        class_dir = os.path.join(test_dir, 'normal')
    elif label == 1:
        class_dir = os.path.join(test_dir, 'MSIplus')
    else:
        class_dir = os.path.join(test_dir, 'MSIminus')
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f'image_{i}.jpg')
    # Save image to the appropriate class directory
    # Assuming image is a PIL image
    # image.save(image_path)
    cv2.imwrite(image_path, image)

#!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/

if(torch.cuda.is_available()):
    print("CUDA is present")
    print(torch.cuda.current_device())
    torch.cuda.get_device_name(0)
else:
    print("CUDA is absent")

#%pip install -r requirements.txt

# Train YOLOv5s Classification on Imagenette160 for 3 epochs
#!python classify/train.py --model yolov5s-cls.pt --data ./tiles/train --epochs 5 --img 224 --cache
# python classify/train.py --model yolov5s-cls.pt --data D:\YASHWANTH\yolov5\tiles\ --epochs 1000 --img 1024 --cache disk --batch-size 4
#!python predict.py ./_.ndpi
