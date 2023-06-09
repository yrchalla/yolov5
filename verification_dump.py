# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:05:08 2023
@author: Lucid
"""
# import os, sys
# from lazy_import import lazy_module

# if getattr(sys, 'frozen', False):
#     # The application is running as a bundled executable
#     current_dir = os.path.dirname(sys.executable)
# else:
#     # The application is running as a script
#     current_dir = os.path.dirname(os.path.abspath(__file__))

# OPENSLIDE_PATH = os.path.join(current_dir, 'openslide-win64-20230414', 'bin')
# # if hasattr(os, 'add_dll_directory'):
# #     # Python >= 3.8 on Windows
# #     with os.add_dll_directory(OPENSLIDE_PATH):
# #         openslide = lazy_module("openslide")
# # else:
# #     openslide = lazy_module("openslide")

# os.add_dll_directory(OPENSLIDE_PATH)
# # openslide = lazy_module("openslide")
# import openslide

# ET =lazy_module("xml.etree.ElementTree")
# large_image = lazy_module("large_image")
# np = lazy_module("numpy")
# random = lazy_module("random")
# plt = lazy_module("matplotlib.pyplot")
# cv2 = lazy_module("cv2")
# pd = lazy_module("pandas")
# imsave = lazy_module("tifffile.imsave")
# from defect_tile_cut import get_lnb, update_SameTile_annotations, tile_intersection

import os, sys, time

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

import xml.etree.ElementTree as ET
import large_image
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tifffile import imsave
from defect_tile_cut import get_lnb, update_SameTile_annotations, tile_intersection

import large_image

def plot_one_box(x, img, color=None, label=None, line_thickness=4):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.004 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def write_tile_title(img, label, color):
    c1, c2 = (50, 50), (200, 200)
    # line/font thickness
    tl = round(0.004 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img


def get_box_list(wsi_path, xml_path, nm_p):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1, y1, x2, y2 = 0, 0, 0, 0
    box_list = []
    X_Reference, Y_Reference = get_referance(wsi_path, nm_p)
    for elem in root.iter():
        # print(elem.tag)
        if elem.tag == 'ndpviewstate':
            title = elem.find('title').text
            cx = int((int(elem.find('x').text) + X_Reference)/nm_p)
            cy = int((int(elem.find('y').text) + Y_Reference)/nm_p)

        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))
                y.append(int(sub.find('y').text))
            x1 = int((min(x) + X_Reference)/nm_p)
            x2 = int((max(x) + X_Reference)/nm_p)
            y1 = int((min(y) + Y_Reference)/nm_p)
            y2 = int((max(y) + Y_Reference)/nm_p)
            row = (title, cx, cy, x1, y1, x2, y2)
            box_list.append(row)
    return box_list


def get_referance(wsi_path, nm_p):
    slide = slideRead(wsi_path)

    w = int(slide.properties.get('openslide.level[0].width'))
    h = int(slide.properties.get('openslide.level[0].height'))

    ImageCenter_X = (w/2)*nm_p
    ImageCenter_Y = (h/2)*nm_p

    OffSet_From_Image_Center_X = slide.properties.get(
        'hamamatsu.XOffsetFromSlideCentre')
    OffSet_From_Image_Center_Y = slide.properties.get(
        'hamamatsu.YOffsetFromSlideCentre')

    # print("offset from Img center units?", OffSet_From_Image_Center_X,OffSet_From_Image_Center_Y)

    X_Ref = float(ImageCenter_X) - float(OffSet_From_Image_Center_X)
    Y_Ref = float(ImageCenter_Y) - float(OffSet_From_Image_Center_Y)

    # print(ImageCenter_X,ImageCenter_Y)
    # print(X_Reference,Y_Reference)
    return X_Ref, Y_Ref


# def grid_image(folder):
#     imageList =  os.listdir(folder)

def grid_image(imageList, dump_folder, filename, grid_dim, nos_tiles=9):
    assert nos_tiles == grid_dim[0]*grid_dim[1]
    n = len(imageList)
    if n % nos_tiles == 0.0:
        grids = int(n / nos_tiles)
    else:
        grids = int(n / nos_tiles) + 1
    flag = False
    print('total_tiles, nos_tile per grid, grids', n, nos_tiles, grids)
    np_empty = np.zeros([100, 100, 3])
    for grid in range(grids):
        # n_x_y = int(sqrt(nos_tiles))
        for i in range(nos_tiles):
            plt.subplot(grid_dim[0], grid_dim[1], i + 1)
            plt.axis("off")
            # plt.autoscale(tight = True)
            # img = imread(os.path.join(folder,imageList[grid*9+i]))
            if grid*nos_tiles + i < n:
                plt.imshow(imageList[grid*nos_tiles + i], aspect='auto')
            else:
                plt.imshow(np_empty, aspect='auto')
        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        f_path = os.path.join(
            dump_folder, '{0}grid{1}.tif'.format(filename, grid))
        plt.tight_layout()
        # plt.margins(x=0.0,y=0.0, tight=True)
        plt.savefig(f_path, dpi=350)
        # plt.imsave(fname, arr, kwargs)
        plt.show()
    print('done dumping grid @', dump_folder)

# plt(figsize=(20, 20), dpi=200)


def update_SameTile_annotes(gt_list, tile, anote):

    for gt in gt_list:
        breath = abs(gt[5]-gt[3])
        length = abs(gt[6]-gt[4])
        cx = int((gt[3]+gt[5])/2)  # int(gt[1])
        cy = int((gt[4]+gt[6])/2)  # int(gt[2])
        anote_box = (cx-breath/2, cy-length/2, cx+breath/2, cy+length/2)
        if (tile_intersection(tile, anote_box)):
            x1 = anote_box[0]-tile[0]
            y1 = anote_box[1]-tile[1]
            box = (int(x1), int(y1), int(x1+breath), int(y1+length))
            # print(box/1024)
            anote.append(box)
    return anote


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_tile_dims(cx, cy, breath, length, dims, tile_size):
    randx, randy = 0, 0
    left, top = -1, -1

    while (left < 0 or left > dims[0]-tile_size or top < 0 or top > dims[1]-tile_size):
        randx = random.randint(100, 300)
        randy = random.randint(100, 300)
        if cx >= 100 + breath/2:
            left = int(cx - randx - breath/2)
        else:
            left = 0
            randx = 0
        if cy > 100 + length/2:
            top = int(cy - randy - length/2)
        else:
            top = 0
            randy = 0
    return randx, randy, left, top


def plot_anotes(np_img, tile_anote, predict_box_list, left, top, dump, title):
    flag = False
    for k in range(len(tile_anote)):
        # draw ground truthon image
        np_img = plot_one_box(tile_anote[k], np_img, color=(0, 0, 255))
        for pred in predict_box_list:
            p_box = (pred[3]-left, pred[4]-top, pred[5]-left, pred[6]-top)
            if bb_intersection_over_union(tile_anote[k], p_box) > 0.2:
                flag = True
                # draw the predict
                np_img = plot_one_box(p_box, np_img, color=(0, 255, 0))
                # break
        if not flag:
            print('check', title)
            np_img = write_tile_title(np_img, title, color=(255, 255, 255))
            imsave(os.path.join(dump, title + '.tif'), np_img)
    return np_img, flag


def plot_anotes_in_tile(np_img, tile_anote, predict_list, gt_list, left, top):
    flag = False
    for gt in gt_list:
        gt_box = (gt[3]-left, gt[4]-top, gt[5]-left, gt[6]-top)
        if bb_intersection_over_union(tile_anote, gt_box) > 0:
            # if tile_intersection(tile,gt_box):
            # flag = True
            # print('yes_gt')
            # draw the predict
            np_img = plot_one_box(gt_box, np_img, color=(0, 0, 255))

    for pred in predict_list:
        p_box = (pred[3]-left, pred[4]-top, pred[5]-left, pred[6]-top)
        if bb_intersection_over_union(tile_anote, p_box) > 0:
            # if tile_intersection(tile, p_box):
            print('yes_pred')
            # flag = True
            # draw the predict
            np_img = plot_one_box(p_box, np_img, color=(0, 255, 0))

    return np_img, flag


def dump_verification_tiles_2(folder, filename, dump_folder, tile_size=512, nm_p=221):

    # reading all files

    wsi_path = os.path.join(folder, filename + '.ndpi')

    verification_dump_path = os.path.join(folder, 'dump')

    gt_xml_path = os.path.join(folder, filename + '.ndpi.ndpa')

    predict_xml_path = os.path.join(folder, filename + '_predicts.xml')

    df = pd.read_csv(os.path.join(folder, filename + '.csv'))

    # Get annotation info in pixels & dereferenced
    # gt_list = get_box_list(wsi_path,gt_xml_path,nm_p)
    lnb = get_lnb(gt_xml_path, nm_p)
    predict_box_list = get_box_list(wsi_path, predict_xml_path, nm_p)

    # get tile source

    ts = large_image.getTileSource(wsi_path)
    slide = openslide.open_slide(wsi_path)
    dims = slide.dimensions
    # print(dims)

    # initialize
    tp_count = 0
    count = 0

    # loop through the ground truth to get list of tile imgs
    tile_list = []
    # for gt in gt_list:
    for index, row in df.iterrows():
        tile_anote = []

        # recover annotations
        cx, cy = int(row.iloc[3]), int(row.iloc[4])
        length, breath = lnb[index][1], lnb[index][2]
        title = row.iloc[2]
        print('{0}_{1}.tif'.format(title, count))
        # print('cx,cy',cx,cy)
        x1, y1, left, top = get_tile_dims(
            cx, cy, breath, length, dims, tile_size)
        # print(left,top)
        # print(breath,length)
        tile_box = (left, top, left + tile_size, top + tile_size)
        # loop over df if any annotations inside the tile
        # box = (int(x1), int(randy), int(randx+breath), int(randy+length))
        tile_anote = update_SameTile_annotations(df, tile_box, tile_anote, lnb)
        # print(tile_anote)

        im_roi, _ = ts.getRegion(
            region=dict(left=left, top=top, width=tile_size,
                        height=tile_size, units='base_pixels'),
            scale=dict(magnification=40),
            format=large_image.tilesource.TILE_FORMAT_PIL)

        file = im_roi.convert('RGB')
        np_img = np.array(file)
        # print(np_img.shape)

        np_img, flag = plot_anotes(
            np_img, tile_anote, predict_box_list, left, top)
        if flag:
            tp_count += 1
        filepath = os.path.join(
            dump_folder, '{0}_{1}.tif'.format(title, count))
        tile_list.append(np_img)
        # imsave(filepath,np_img)
        count += 1
        # if count > 5 : break
    fp_count = (len(predict_box_list)-tp_count)
    tp_rate = tp_count/len(lnb)
    fp_rate = fp_count/len(predict_box_list)
    print('tp_count', tp_count, 'recall or tp rate :', tp_rate)
    print('fp_count', fp_count, 'precsion or fp rate', fp_rate)
    return tile_list


"""
without using csv

"""


def dump_verification_tiles(folder, filename, dump_folder, tile_size=512, nm_p=221):

    wsi_path = os.path.join(folder, filename + '.ndpi')

    # verification_dump_path = os.path.join(folder,'dump')

    gt_xml_path = os.path.join(folder, filename + '.ndpi.ndpa')

    predict_xml_path = os.path.join(folder, filename + '_predicts_pruned1.xml')
    print(predict_xml_path)
    # Get annotation info in pixels & dereferenced
    gt_list = get_box_list(wsi_path, gt_xml_path, nm_p)
    predict_list = get_box_list(wsi_path, predict_xml_path, nm_p)

    # get tile source

    ts = large_image.getTileSource(wsi_path)
    slide = openslide.open_slide(wsi_path)
    dims = slide.dimensions
    # print(dims)

    # initialize
    tp_count = 0
    count = 0

    # loop through the ground truth to get list of tile imgs
    tile_list = []

    # create folder if they don't exist

    # tp_folder = os.path.join(dump_folder,'tps')
    # if not os.path.isdir(tp_folder) : os.mkdir(tp_folder)
    tn_folder = os.path.join(dump_folder, 'tns')
    if not os.path.isdir(tn_folder):
        os.mkdir(tn_folder)

    # loop through the ground truth
    count = 0
    for gt in gt_list:
        tile_anote = []

        title = gt[0]
        # centre of annotation in pixels
        cx = int((gt[3]+gt[5])/2)  # int(gt[1])
        cy = int((gt[4]+gt[6])/2)  # int(gt[2])

        # breath in pixels

        breath = abs(gt[5]-gt[3])
        length = abs(gt[6]-gt[4])

        # x1,y1,left,top = get_tile_dims(cx, cy, breath, length, dims, tile_size)
        # centering the Groundtruth

        x1, y1 = tile_size/2, tile_size/2

        left = int(cx - x1 - breath/2)
        top = int(cy - y1 - breath/2)
        # print(left,top)

        # print(breath,length)
        tile_box = (left, top, left+tile_size, top + tile_size)
        # gt_box = (gt[3]-left, gt[4]-top, gt[5] - left ,gt[6] - top)
        # gt_box = (x1, y1, x1+breath, y1+length)
        # print(gt_box)

        tile_anote = update_SameTile_annotes(gt_list, tile_box, tile_anote)

        im_roi, _ = ts.getRegion(
            region=dict(left=left, top=top, width=tile_size,
                        height=tile_size, units='base_pixels'),
            scale=dict(magnification=40),
            format=large_image.tilesource.TILE_FORMAT_PIL)

        # print(im_roi.shape)
        file = im_roi.convert('RGB')
        np_img = np.array(file)
        # print(np_img.shape)
        tile_box = (0, 0, tile_size, tile_size)
        title = title + '_{0}'.format(count)
        # dumps_tns
        np_img, flag = plot_anotes(
            np_img, tile_anote, predict_list, left, top, tn_folder, title)
        # print(len(predict_box_list))
        # np_img, flag  = plot_anotes_in_tile(np_img, tile_box, predict_box_list,gt_list,left, top)
    #     if flag: tp_count += 1
    #     #filepath = os.path.join(tp_folder,filename +'_{0}_{1}.tif'.format(title,count))
    #     #imsave(filepath,np_img)
        np_img = write_tile_title(np_img, title, color=(255, 255, 255))
        tile_list.append(np_img)
        count += 1
    #     #if count > 15 : break
    return tile_list


def dump_fp_tiles(folder, filename, dump_folder, tile_size=512, nm_p=221):

    wsi_path = os.path.join(folder, filename + '.ndpi')

    gt_xml_path = os.path.join(folder, filename + '.ndpi.ndpa')

    predict_xml_path = os.path.join(folder, filename + '_predicts.xml')

    # Get annotation info in pixels & dereferenced
    gt_list = get_box_list(wsi_path, gt_xml_path, nm_p)
    predict_list = get_box_list(wsi_path, predict_xml_path, nm_p)

    # get tile source

    ts = large_image.getTileSource(wsi_path)
    slide = openslide.open_slide(wsi_path)
    dims = slide.dimensions
    # print(dims)

    # initialize
    tp_count = 0
    count = 0

    # loop through the ground truth to get list of tile imgs
    tile_list = []

    # create folder if they don't exist

    fp_folder = os.path.join(dump_folder, 'fps')
    if not os.path.isdir(fp_folder):
        os.mkdir(fp_folder)
    # loop through the ground truth
    count = 0
    for pred in predict_list:

        tile_anote = []

        # print(gt)
        title = pred[0]
        # centre of annotation in pixels
        cx = int((pred[3]+pred[5])/2)  # int(gt[1])
        cy = int((pred[4]+pred[6])/2)  # int(gt[2])

        # breath in pixels

        breath = abs(pred[5]-pred[3])
        length = abs(pred[6]-pred[4])

        # centering the Groundtruth

        x1, y1 = tile_size/2, tile_size/2

        left = int(cx - x1 - breath/2)
        top = int(cy - y1 - breath/2)
        # print(left,top)

        # print(breath,length)
        tile_box = (left, top, left+tile_size, top + tile_size)
        # gt_box = (gt[3]-left, gt[4]-top, gt[5] - left ,gt[6] - top)
        # gt_box = (x1, y1, x1+breath, y1+length)
        # print(gt_box)

        tile_anote = update_SameTile_annotes(
            predict_list, tile_box, tile_anote)

        im_roi, _ = ts.getRegion(
            region=dict(left=left, top=top, width=tile_size,
                        height=tile_size, units='base_pixels'),
            scale=dict(magnification=40),
            format=large_image.tilesource.TILE_FORMAT_PIL)

        # print(im_roi.shape)
        file = im_roi.convert('RGB')
        np_img = np.array(file)
        # print(np_img.shape)
        tile_box = (0, 0, tile_size, tile_size)
        # check anote is fp
        for k in range(len(tile_anote)):
            for gt in gt_list:
                gt_box = (gt[3]-left, gt[4]-top, gt[5]-left, gt[6]-top)
                if bb_intersection_over_union(tile_anote[k], gt_box) <= 0:
                    # draw the predict
                    np_img = plot_one_box(
                        tile_anote[k], np_img, color=(0, 255, 0))
        # filepath = os.path.join(tp_folder,filename +'_{0}_{1}.tif'.format(title,count))
        # imsave(filepath,np_img)
        np_img = write_tile_title(
            np_img, title + '{0}'.format(count), color=(255, 255, 255))
        tile_list.append(np_img)
        count += 1
        # if count > 15 : break
    return tile_list


################# YASHWANTH

def get_id_box_list(wsi_path, xml_path, nm_p):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1, y1, x2, y2 = 0, 0, 0, 0
    box_list = []
    id_list = []    # MOD
    X_Reference, Y_Reference = get_referance(wsi_path, nm_p)
    for elem in root.iter():
        # print(elem.tag)
        if elem.tag == 'ndpviewstate':
            title = elem.find('title').text
            cx = int((int(elem.find('x').text) + X_Reference)/nm_p)
            cy = int((int(elem.find('y').text) + Y_Reference)/nm_p)
            id = elem.get("id")   # MOD

        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))
                y.append(int(sub.find('y').text))
            x1 = int((min(x) + X_Reference)/nm_p)
            x2 = int((max(x) + X_Reference)/nm_p)
            y1 = int((min(y) + Y_Reference)/nm_p)
            y2 = int((max(y) + Y_Reference)/nm_p)
            row = (title, cx, cy, x1, y1, x2, y2)
            box_list.append(row)
            id_list.append(id)  # MOD
    return box_list, id_list    # MOD


def get_np_predicts(folder, filename, tile_list, format, tile_size=512, nm_p=221):

    wsi_path = os.path.join(folder, filename + '.' + format)

    predict_xml_path = os.path.join(folder, filename + '_predicts.xml')

    # Get annotation info in pixels & dereferenced
    predict_list, id_list = get_id_box_list(wsi_path, predict_xml_path, nm_p)

    # get tile source

    # ts = large_image.getTileSource(wsi_path)
    slide = slideRead(wsi_path)
    # dims = slide.dimensions
    # print(dims)

    # initialize
    tp_count = 0
    count = 0

    # loop through the ground truth to get list of tile imgs
    # tile_list = []

    # create folder if they don't exist

    # fp_folder = os.path.join(dump_folder, 'fps')
    # if not os.path.isdir(fp_folder):
    #     os.mkdir(fp_folder)
    # loop through the ground truth
    count = 0
    for pred in predict_list:
        tile_anote = []

        # print(gt)
        title = pred[0]
        # centre of annotation in pixels
        cx = int((pred[3]+pred[5])/2)  # int(gt[1])
        cy = int((pred[4]+pred[6])/2)  # int(gt[2])

        # breath in pixels

        breath = abs(pred[5]-pred[3])
        length = abs(pred[6]-pred[4])

        # centering the Groundtruth

        x1, y1 = tile_size/2, tile_size/2

        left = int(cx - x1)
        top = int(cy - y1)
        # print(left,top)

        # print(breath,length)
        tile_box = (left, top, left+tile_size, top + tile_size)
        # gt_box = (gt[3]-left, gt[4]-top, gt[5] - left ,gt[6] - top)
        # gt_box = (x1, y1, x1+breath, y1+length)
        # print(gt_box)

        tile_anote = update_SameTile_annotes(
            predict_list, tile_box, tile_anote)

        level = slide.get_best_level_for_downsample(1.0 / 40)
        im_roi = slide.read_region((left, top), level, (tile_size, tile_size))
        
        # im_roi = im_roi.convert("RGB")

        # print(im_roi.shape)
        file = im_roi.convert('RGB')
        np_img = np.array(file)
        # print(np_img.shape)
        tile_box = (0, 0, tile_size, tile_size)
        # check anote is fp
        for k in range(len(tile_anote)):
            # draw the predict
            np_img = plot_one_box(
                tile_anote[k], np_img, color=(0, 255, 0))
        # filepath = os.path.join(tp_folder,filename +'_{0}_{1}.tif'.format(title,count))
        # imsave(filepath,np_img)
        np_img = write_tile_title(
            np_img, title + '{0}'.format(count), color=(255, 255, 255))
        tile_list.append(np_img)
        if len(tile_list) % 4 == 0:
            time.sleep(0)
        count += 1
        # if count > 15 : break

def count_predicts(folder, filename, format, tile_size=512, nm_p=221):
    wsi_path = os.path.join(folder, filename + '.' + format)

    predict_xml_path = os.path.join(folder, filename + '_predicts.xml')

    # Get annotation info in pixels & dereferenced
    predict_list, id_list = get_id_box_list(wsi_path, predict_xml_path, nm_p)
    return len(predict_list)

def slideRead(wsi_path):
    if wsi_path.endswith(".ndpi"):
        slide = openslide.open_slide(wsi_path)
    return slide

def get_set(folder, filename):
    xml_path = os.path.join(folder, filename + '_predicts.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ret = set()
    # x1, y1, x2, y2 = 0, 0, 0, 0
    # box_list = []
    # id_list = []    # MOD
    # X_Reference, Y_Reference = get_referance(wsi_path, nm_p)
    for elem in root.iter():
        # print(elem.tag)
        if elem.tag == 'ndpviewstate':
            # title = elem.find('title').text
            # cx = int((int(elem.find('x').text) + X_Reference)/nm_p)
            # cy = int((int(elem.find('y').text) + Y_Reference)/nm_p)
            id = elem.get("id")   # MOD
            fp_tp = elem.find('fp-tp').text
            if (fp_tp == 'fp'):
                ret.add(int(id))
    
    return ret
