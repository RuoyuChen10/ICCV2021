# -*- coding: utf-8 -*-  

"""
Created on 2021/2/10

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

from utils import *
import Datasets.dataload as dl
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from models.BiSeNet.BiSeNet import BiSeNet
from demo import get_last_conv_name,get_heatmap_method

atts = ['background','skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

def Segmentation(image_dir):
    '''
    Segment the face
        image_dir: the input, need the align face
    ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
     'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose',
     'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l',
     'cloth', 'hair', 'hat']
    '''
    # Load BiSeNet network
    net = BiSeNet(n_classes=19)
    # GPU
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load("./pre-trained/BiSeNet.pth"))
    net.eval()
    # Process method
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Evaluation
    with torch.no_grad():
        image = Image.open(image_dir)
        # Resize
        image = image.resize((512, 512), Image.BILINEAR)
        # processing
        image = to_tensor(image)
        image = torch.unsqueeze(image, 0)
        if torch.cuda.is_available():
            image = image.cuda()
        out = net(image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return parsing

def merge_seg_cam(image_dir,heatmap,parsing,types=["left_brow", 'right_brow', 'left_eye', 'right_eye', 'nose', 'mouth']):
    '''
    Merge the face with heatmap aas seg_cam
        image_dir: the input, need the align face
    ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
     'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose',
     'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l',
     'cloth', 'hair', 'hat']
    '''
    # Load image
    image = cv2.imread(image_dir)
    # Size
    size = parsing.shape
    image = cv2.resize(image,size)
    heatmap = cv2.resize(heatmap,size)
    # Type:
    parsing = parsing.astype(np.uint8)

    for typer in types:
        pi = atts.index(typer)
        index = np.where(parsing == pi)
        image[index[0], index[1], :] = 0.5*image[index[0], index[1], :]+0.5*heatmap[index[0], index[1], :]
    return image

def calculate_score(heatmap,parsing,types=["left_brow", 'right_brow', 'left_eye', 'right_eye', 'nose', 'mouth']):
    '''
    Calculate the average score of different part
    '''
    # Size
    size = parsing.shape
    heatmap = cv2.resize(heatmap,size)
    print("Face part score:")
    information = {}
    for typer in types:
        if typer == 'mouth':
            # zeros matrix
            calculate_map = np.zeros(size)
            # which part index
            index1 = np.where(parsing == 12)
            index2 = np.where(parsing == 13)
            calculate_map[index1[0], index1[1]] = heatmap[index1[0], index1[1]]
            calculate_map[index2[0], index2[1]] = heatmap[index2[0], index2[1]]
            # calcute the average:
            score = np.sum(calculate_map) / (np.array(index1).shape[1]+np.array(index2).shape[1])
        else:
            # zeros matrix
            calculate_map = np.zeros(size)
            # which part index
            pi = atts.index(typer)
            index = np.where(parsing == pi)
            # write the value of heatmap to matrix
            calculate_map[index[0], index[1]] = heatmap[index[0], index[1]]
            # calcute the average:
            score = np.sum(calculate_map) / np.array(index).shape[1]
        information[typer] = score
        print("    "+typer+": "+'%.4f' % score)
    return information

def main(args):
    # Read the images as input
    images = dl.Image_precessing(args.image_input)
    inputs = torch.tensor([images], requires_grad=True)
    # Load the networks
    net = get_network(args.network)
    # Heatmap
    cam = get_heatmap_method(net, method=args.heatmap_method)
    # Mask
    mask = cam(inputs.cuda())  # cam mask
    cam.remove_handlers()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask[0]), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    # Empty the CUDA menmory
    torch.cuda.empty_cache()
    net = None
    # Segment Net
    parsing = Segmentation(args.image_input)
    print(np.unique(parsing))
    image = merge_seg_cam(args.image_input,heatmap,parsing)
    
    saver = args.network + "-" + args.heatmap_method + '-' + args.image_input.split('/')[-1]
    cv2.imwrite(os.path.join(args.output_dir,saver),image)
    # calculate the average score
    calculate_score(mask[0],parsing)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-input', type=str, default="./images/0001_01_align.jpg",
                        help='input image path')
    parser.add_argument('--network', type=str, default='VGGFace2',
                        help='Face identity recognition network.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--output-dir', type=str, default='./results/segment-cam',
                        help='output directory to save results')
    args = parser.parse_args()
    
    mkdir(args.output_dir)
    
    main(args)