"""
Predict:         python classify/predict.py --weights runs\train-cls\exp18\weights\best.pt --source im.jpg
Validate:        python classify/val.py --weights runs\train-cls\exp18\weights\best.pt --data D:\YASHWANTH\yolov5\tiles
Export:          python export.py --weights runs\train-cls\exp18\weights\best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs\train-cls\exp18\weights\best.pt')
"""

# write python code to break an ndpi whole slide image into tiles of 1024 pixels and run predict for each tile 
import sys, zipfile, os, requests, re, time
start_time = time.time()
from subprocess import Popen, PIPE, STDOUT
from classify.predict import run_with_prediction
from pathlib import Path
import numpy as np
TILE_SIZE = 1024
NM_P = 221

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

WSI_PATH = sys.argv[1]
if not WSI_PATH.endswith('.ndpi'):
    print("Require NDPI")
    sys.exit()

heatmapPrefix = '<?xml version="1.0" encoding="utf-8" standalone="yes"?><annotations>'
heatmapSuffix = '</annotations>'
slide = slideRead(WSI_PATH)
LEVEL = slide.get_best_level_for_downsample(1.0 / 40)
X_Reference, Y_Reference = get_referance(WSI_PATH, NM_P)
slide_width, slide_height = slide.dimensions
id = 0
ims = []
for i in range(int(slide_width / TILE_SIZE)):
    for j in range(int(slide_height / TILE_SIZE)):
        print(id , '/' , int(slide_width / TILE_SIZE)*int(slide_height / TILE_SIZE))
        id+=1
        im_roi = slide.read_region((TILE_SIZE * i, j * TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
        im_roi = im_roi.convert("RGB")
        im_roi = np.array(im_roi)
        im_roi = im_roi.transpose(2,0,1)    # 1024, 1024, 3 to 3, 1024, 1024
        ims.append(im_roi)
        print(im_roi.shape)

preds = run_with_prediction('best.pt', ims)
print(len(preds))
id = 0
for i in range(int(slide_width / TILE_SIZE)):
    for j in range(int(slide_height / TILE_SIZE)):
        id+=1
        pred = preds[id-1]
        cx = (i*TILE_SIZE+TILE_SIZE/2)*NM_P - X_Reference
        cy = (j*TILE_SIZE+TILE_SIZE/2)*NM_P - Y_Reference

        if (pred == 'normal'):
            color = '#0000ff'
            colorname = 'blue'
        else:
            color = '#ff0000'
            colorname = 'red'
        heatmapPrefix += f"""<ndpviewstate id="{id}">
                <title>{pred}</title>
                <details/>
                <coordformat>nanometers</coordformat>
                <lens>4.630434</lens>
                <x>1334342</x>
                <y>2423208</y>
                <z>0</z>
                <showtitle>0</showtitle>
                <showhistogram>0</showhistogram>
                <showlineprofile>0</showlineprofile>
                <annotation type="pin" displayname="AnnotatePin" color="{color}">
                    <x>{cx}</x>
                    <y>{cy}</y>
                    <icon>pin{colorname}</icon>
                    <stricon>iconpin{colorname}</stricon>
                </annotation>
            </ndpviewstate>
        """
        

with open(sys.argv[1] + '.ndpa', "w") as file:
    # Write the string to the file
    file.write(heatmapPrefix+heatmapSuffix)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time / 60} min")
