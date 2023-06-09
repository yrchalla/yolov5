# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:04:18 2022

@author: lucid
"""

"""
the CSV file contains annotations for defects. The annotations are in the form of bounding boxes, which are described by the x and y coordinates of the top-left corner of the box, as well as its width and height. The bounding boxes correspond to the locations of the defects within the scanned image.
"""
# import os, sys
# from lazy_import import lazy_module

# # current_dir = os.path.dirname(os.path.abspath(__file__))

# if getattr(sys, 'frozen', False):
#     # The application is running as a bundled executable
#     current_dir = os.path.dirname(sys.executable)
# else:
#     # The application is running as a script
#     current_dir = os.path.dirname(os.path.abspath(__file__))

# OPENSLIDE_PATH = os.path.join(current_dir, 'openslide-win64-20230414', 'bin')
# os.add_dll_directory(OPENSLIDE_PATH)
# openslide = lazy_module("openslide")

# large_image = lazy_module("large_image")
# imsave = lazy_module("tifffile.imsave")
# Writer = lazy_module("pascal_voc_writer.Writer")
# pd = lazy_module("pandas")
# random = lazy_module("random")
# ET = lazy_module("xml.etree.ElementTree")

import os, sys

# current_dir = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    # The application is running as a bundled executable
    current_dir = os.path.dirname(sys.executable)
else:
    # The application is running as a script
    current_dir = os.path.dirname(os.path.abspath(__file__))

OPENSLIDE_PATH = os.path.join(current_dir, 'openslide-win64-20230414', 'bin')
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import large_image
import os
from tifffile import imsave
from pascal_voc_writer import Writer
import pandas as pd
import random
import xml.etree.ElementTree as ET


def get_lnb(xml_path, nm_p=221):
    """
    Parse the xml file containing the defects, and store the id, length and breadth of each defect in the lnb array. the defect's id is the id of its ancestor ndpvewstate element. the xml parse is a dfs
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1, y1, x2, y2 = 0, 0, 0, 0
    lnb = []
    for elem in root.iter():
        # print(elem.tag)
        if elem.tag == 'ndpviewstate':
            _id = elem.attrib.get('id')
        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))
                y.append(int(sub.find('y').text))
            x1 = int(min(x)/nm_p)
            x2 = int(max(x)/nm_p)
            y1 = int(min(y)/nm_p)
            y2 = int(max(y)/nm_p)
            breath = abs(x2-x1)
            length = abs(y2-y1)
            row = (int(_id), breath, length)
            lnb.append(row)
    return lnb

# def handle_border_tiles(left,top,tile_size,x1,y1,breath,length):
#     if (left + tile_size) >= dims[0]:
#           left  = dims[0]-tile_size - breath/2
#     if left < 0 :
#           left = 0
#     if(top +tile_size)>= dims[1]:
#           top = dims[1]-tile_size - length/2
#     if (top) < 0:
#           top = 0
#     return left,top


def tile_intersection(boxA, boxB, limit=50):
    """
    The tile_intersection() function is used to determine if a tile overlaps with an annotation. If the intersection of the two boxes has an area greater than a certain limit, it is classified as an overlap.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return (interArea > limit)


def update_SameTile_annotations(df, tile, anote, lnb):

    for i, row in df.iterrows():
        breath = lnb[i][1]
        length = lnb[i][2]
        cx = int(row.iloc[3])
        cy = int(row.iloc[4])
        anote_box = (cx-breath/2, cy-length/2, cx+breath/2, cy+length/2)
        if (tile_intersection(tile, anote_box)):
            x1 = anote_box[0]-tile[0]
            y1 = anote_box[1]-tile[1]
            box = (int(x1), int(y1), int(x1+breath), int(y1+length))
            # print(box/1024)
            anote.append(box)
    return anote


def check_tile4annotations(df, tile, anote, lnb):

    for i, row in df.iterrows():
        breath = lnb[i][1]
        length = lnb[i][2]
        cx = int(row.iloc[3])
        cy = int(row.iloc[4])
        anote_box = (cx-breath/2, cy-length/2, cx+breath/2, cy+length/2)
        if (tile_intersection(tile, anote_box)):
            x1 = anote_box[0]-tile[0]
            y1 = anote_box[1]-tile[1]
            box = (int(x1), int(y1), int(x1+breath), int(y1+length))
            # print(box/1024)
            anote.append(box)
    return anote


def Get_Defect_tiles(fname, lnb, csv_path, dump_folder, ts, dims, defect, tile_size=1024):
    """
    generate some tiles having defects. by randomly perturbong it around each defecct
    select random tiles from the image and check intersection with defect tiles. if intersection is more than a threshold, classify the randomly chosen tile as a defect tile and The coordinates of the defect tile are stored in a separate CSV file.
    """
    df = pd.read_csv(csv_path)
    # print(df.head())
    count = 0
    images_folder = os.path.join(dump_folder, 'images')
    labels_folder = os.path.join(dump_folder, 'labels')
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
    if not os.path.isdir(labels_folder):
        os.mkdir(labels_folder)

    print_buffer = []
    for index, row in df.iterrows():
        # for each defect
        tile_anote = []
        for i in range(1, 2):
            randx = random.randint(200, 700)
            randy = random.randint(200, 700)

            # pascal_voc

            x1 = randx
            y1 = randy
            breath = lnb[index][1]
            length = lnb[index][2]
            # print(breath, length)

            cx = int(row.iloc[3])
            cy = int(row.iloc[4])
            # print(cx,cy)
            left = cx - randx - breath/2
            top = cy - randy - length/2
            tile_box = (left, top, left + tile_size, top + tile_size)
            # loop over df if any annotations inside the tile
            box = (int(x1), int(y1), int(x1+breath), int(y1+length))
            # tile_anote.append(box)
            tile_anote = update_SameTile_annotations(
                df, tile_box, tile_anote, lnb)
            # handle border tiles cutting - ignoring as its just one or two tiles.
            # print(tile_anote.dtype)
            # left,top, x1, y1 = handle_border_tiles(left,top,tile_size,x1,y1,breath,length)

            im_roi, _ = ts.getRegion(
                region=dict(left=left, top=top, width=tile_size,
                            height=tile_size, units='base_pixels'),
                scale=dict(magnification=40),
                format=large_image.tilesource.TILE_FORMAT_NUMPY)

            filename = r'{0}_{1}_{2}.tif'.format(fname, row.iloc[0], i)
            if im_roi.shape[0] != tile_size or im_roi.shape[1] != tile_size:
                print('In', fname, 'tile:', filename, 'is abnormal')
            else:

                filepath = os.path.join(images_folder, filename)
                writer = Writer(filepath, tile_size, tile_size)
                imsave(filepath, im_roi)
                count += 1
                if defect:
                    if len(tile_anote) > 1:
                        print(filename, " : ", len(tile_anote))
                    # ndpi-tile origin filename and left, top
                    # Then later we open labels and add the relative position yolo*tile-size and get it back to xml-ndpa
                    # run through csv - open yolotext, add and then write into ndpa format
                    # print("{} {:.3f} {:.3f}".format(filename, left,top))
                    print_buffer.append((filename, left, top))
                    for k in range(len(tile_anote)):
                        # print("abnormal",tile_anote[k][0],tile_anote[k][1], tile_anote[k][2], tile_anote[k][3])
                        # writer.addObject("abnormal",int(x1), int(y1), int(x1+breath), int(y1+length))
                        writer.addObject(
                            "abnormal", tile_anote[k][0], tile_anote[k][1], tile_anote[k][2], tile_anote[k][3])
                    xml_name = os.path.join(
                        labels_folder, filename.replace("tif", "xml"))
                    # print(xml_name)
                    writer.save(xml_name)  # save pascal_voc annotation
                else:
                    yolo_name = os.path.join(
                        labels_folder, filename.replace("tif", "txt"))
                    print("\n".join(print_buffer), file=open(yolo_name, "w"))

        count += 1
        if count > 500:
            break
    if defect:
        tile_ref_folder = os.path.join(dump_folder, 'tile_refs')
        save_ndpi_ref_file = os.path.join(tile_ref_folder, fname+'.csv')
        ndpi_df = pd.DataFrame(print_buffer, columns=[
                               'filename', 'left', 'top'])
        ndpi_df.to_csv(save_ndpi_ref_file)


def Get_fp_tiles(fname, lnb, csv_path, dump_folder, ts, dims, defect, tile_size=1024):
    """
    Get false positve tiles
    """
    df = pd.read_csv(csv_path)
    # print(df.head())
    count = 0
    images_folder = os.path.join(dump_folder, 'images')
    labels_folder = os.path.join(dump_folder, 'labels')
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
    if not os.path.isdir(labels_folder):
        os.mkdir(labels_folder)
    print_buffer = []
    df = df.dropna(subset=['Details'])
    print(df['Details'].value_counts())
    for index, row in df.iterrows():
        if not (str(row['Details']) == 'lsiltype' or str(row['Details']) == 'reactive inter'):
            print(str(row['Details']))
            for i in range(1, 2):
                randx = random.randint(200, 700)
                randy = random.randint(200, 700)

                breath = lnb[index][1]
                length = lnb[index][2]
                # print(breath, length)

                cx = int(row.iloc[4])
                cy = int(row.iloc[5])
                # print(cx,cy)
                left = cx - randx - breath/2
                top = cy - randy - length/2

                im_roi, _ = ts.getRegion(
                    region=dict(left=left, top=top, width=tile_size,
                                height=tile_size, units='base_pixels'),
                    scale=dict(magnification=40),
                    format=large_image.tilesource.TILE_FORMAT_NUMPY)

                filename = r'n{0}_{1}_{2}.tif'.format(fname, row.iloc[0], i)
                if im_roi.shape[0] != tile_size or im_roi.shape[1] != tile_size:
                    print('In', fname, 'tile:', filename, 'is abnormal')
                else:
                    filepath = os.path.join(images_folder, filename)
                    # writer = Writer(filepath, tile_size, tile_size)
                    imsave(filepath, im_roi)
                    count += 1
                    yolo_name = os.path.join(
                        labels_folder, filename.replace("tif", "txt"))
                    print("\n".join(print_buffer), file=open(yolo_name, "w"))
            count += 1
        # if count > 500 : break
    print('total tiles :', count)


def dump_annotation_tiles(folder, tile_size, nm_p):
    dump_folder = os.path.join(folder, "dump")
    if not os.path.isdir(dump_folder):
        os.mkdir(dump_folder)

    for files in os.listdir(folder):
        if files.endswith('.ndpi'):
            filename = files.split('.')[0]
            print('start =>', filename)
            csv_name = filename + ".csv"
            csv_path = os.path.join(folder, csv_name)
            # print("csv path :", csv_path)
            # df = pd.read_csv(csv_path)
            xml_name = filename + ".ndpi.ndpa"
            xml_path = os.path.join(folder, xml_name)
            wsi = filename + ".ndpi"
            wsi_path = os.path.join(folder, wsi)
            slide = openslide.open_slide(wsi_path)
            dims = slide.dimensions
            ts = large_image.getTileSource(wsi_path)
            lnb = get_lnb(xml_path, nm_p)
            print('tile dumping starts')
            # Get_Defect_tiles(filename,lnb,csv_path,dump_folder,ts,dims,defect=True,tile_size=1024)
            Get_fp_tiles(filename, lnb, csv_path, dump_folder,
                         ts, dims, defect=True, tile_size=1024)
            print('done =>', filename)
            print('\n')

# old code

# def Get_Defect_tiles(fname,lnb,csv_path,dump_folder,ts,dims,tile_size=1024):
#     count = 1
#     with open(csv_path) as file:
#         #print_buffer = []
#         for line in file:
#             line_list = line.split(",")
#             #print(line)

#             for i in range(1,4):
#                 randx = random.randint(200, 700)
#                 randy = random.randint(200, 700)
#                 #pascal_voc

#                 x1 = randx
#                 y1 = randy

#                 index = int(line_list[0])
#                 breath = lnb[index][1]
#                 length = lnb[index][2]
#                 #print(breath, length)


#                 cx = int(line_list[3])
#                 cy = int(line_list[4])
#                 left = cx -randx -breath/2
#                 top =  cy -randy -length/2

#                 #loop over df if any annotations inside the tile


#                 # handle border tiles cutting

#                 #left,top, x1, y1 = handle_border_tiles(left,top,tile_size,x1,y1,breath,length)

#                 im_roi, _ = ts.getRegion(
#                     region=dict(left= left, top= top, width=tile_size, height=tile_size, units='base_pixels'),
#                     scale=dict(magnification=40),
#                     format=large_image.tilesource.TILE_FORMAT_NUMPY )

#                 filename = r'{0}_{1}_{2}.tif'.format(fname,line_list[0],i)
#                 if im_roi.shape[0]!=tile_size or im_roi.shape[1]!=tile_size:
#                     print('In',fname,'tile:',filename, 'is abnormal')
#                 else :
#                     filepath = os.path.join(dump_folder,filename)
#                     imsave(filepath,im_roi)
#                     writer = Writer(filepath, tile_size, tile_size)
#                     writer.addObject("abnormal",int(x1), int(y1), int(x1+breath), int(y1+length))
#                     xml_name = os.path.join(dump_folder,filename.replace("tif","xml"))
#                     #print(xml_name)
#                     writer.save(xml_name) # save pascal_voc annotation
#             # count +=1
#             # if count > 5 : break
