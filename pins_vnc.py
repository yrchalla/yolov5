import xml.etree.ElementTree as ET

import os, sys, time
import numpy as np
import random, zipfile, requests

from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score

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



# Define a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the transforms for preprocessing
transform = ToTensor()

# Create dataset objects
train_dataset = ImageDataset(X_train, y_train, transform)
val_dataset = ImageDataset(X_val, y_val, transform)
test_dataset = ImageDataset(X_test, y_test, transform)

# Create dataloaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the ResNet-50 model
model = resnet50(pretrained=False)
num_classes = 3  # Number of output classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy += accuracy_score(predicted.cpu(), labels.cpu())

    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_accuracy += accuracy_score(predicted.cpu(), labels.cpu())

    # Calculate average losses and accuracies
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_accuracy / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    val_accuracy = val_accuracy / len(val_dataset)

    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Testing
model.eval()
test_accuracy = 0.0
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_accuracy += accuracy_score(predicted.cpu(), labels.cpu())

test_accuracy = test_accuracy / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

