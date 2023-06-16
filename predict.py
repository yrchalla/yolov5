"""
Predict:         python classify/predict.py --weights runs\train-cls\exp18\weights\best.pt --source im.jpg
Validate:        python classify/val.py --weights runs\train-cls\exp18\weights\best.pt --data D:\YASHWANTH\yolov5\tiles
Export:          python export.py --weights runs\train-cls\exp18\weights\best.pt --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs\train-cls\exp18\weights\best.pt')
"""

# write python code to break an ndpi whole slide image into tiles of 1024 pixels and run predict for each tile 
import sys, zipfile, os, requests, re
from subprocess import Popen, PIPE, STDOUT
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
for i in range(int(slide_width / TILE_SIZE)):
    for j in range(int(slide_height / TILE_SIZE)):
        id+=1
        im_roi = slide.read_region((TILE_SIZE * i, j * TILE_SIZE), LEVEL, (TILE_SIZE, TILE_SIZE))
        im_roi = im_roi.convert("RGB")
        im_roi.save("deleteMe.jpg", "JPEG")
        command = 'python classify/predict.py --weights best.pt --source deleteMe.jpg'
        # result = subprocess.run(command, shell=True, capture_output=True, text=True)\
        # stdout = result.stdout
        process = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        # Get the stdout output as a string
        stdout = process.stdout.read()
        stdout = str(stdout)
        normal_pattern = r"normal\s+([0-9.]+)"
        normal_match = re.search(normal_pattern, stdout)
        normal = float(normal_match.group(1))
        plus_pattern = r"MSIplus\s+(\w+)"
        plus_match = re.search(plus_pattern, stdout)
        plus = float(plus_match.group(1))
        minus_pattern = r"MSIminus\s+(\w+)"
        minus_match = re.search(minus_pattern, stdout)
        minus = float(minus_match.group(1))

        cx = (i*TILE_SIZE+TILE_SIZE/2)*NM_P - X_Reference
        cy = (j*TILE_SIZE+TILE_SIZE/2)*NM_P - Y_Reference

        if (normal == max(normal, plus, minus)):
            color = '#0000ff'
            pred = 'NORMAL'
        elif plus == max(normal, plus, minus):
            color = '#ff0000'
            pred = 'MSI POSITIVE'
        else:
            color = '#ff0000'
            pred = 'MSI NEGATIVE'

        heatmapPrefix += f"""<ndpviewstate id="{id}">
                <title>{pred + ':' + str(max(normal, plus, minus))}</title>
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
                    <icon>pinblue</icon>
                    <stricon>iconpinblue</stricon>
                </annotation>
            </ndpviewstate>
        """

with open(sys.argv[1] + '.ndpa', "w") as file:
    # Write the string to the file
    file.write(heatmapPrefix+heatmapSuffix)
