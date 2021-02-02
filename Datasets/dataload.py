# -*- coding: utf-8 -*-  

"""
Created on 2021/1/31

@author: Ruoyu Chen
"""

import os
import cv2
import numpy as np

def data_dir_read(path):
    '''
    Read the TXT file and get the dictionaries and labels
        path: the path to datasets dir
    '''
    data = []
    for line in open(path):
        data.append(line)
    return data

def Image_precessing(image_dir):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    try:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(image_dir)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return image
    except:
        return None

def analysis_data(data, datasets_dir):
    '''
    Transpose the charactor to image and label
    '''
    inputs = []
    labels1 = []
    labels2 = []
    labels3 = []
    for charactor in data:
        image_dir = charactor.split(' ')[0]
        Image = Image_precessing(os.path.join(datasets_dir,image_dir))
        if Image is not None:
            inputs.append(Image)
            labels1.append(int(charactor.split(' ')[1]))
            labels2.append(int(charactor.split(' ')[2]))
            labels3.append(int(charactor.split(' ')[3][:-1]))
        else:
            continue
    return np.array(inputs), labels1, labels2, labels3

