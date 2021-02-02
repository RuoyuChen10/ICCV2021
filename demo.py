# -*- coding: utf-8 -*-  

"""
Created on 2021/2/3

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np

import Datasets.dataload as dl
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from utils import *

def get_last_conv_name(net):
    """
    Get the name of last convolutional layer
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def get_heatmap_method(net, method="GradCAM"):
    '''
    Get the method to generate heatmap
    '''
    layer_name = get_last_conv_name(net)
    if method == "GradCAM": 
        cam = GradCAM(net, layer_name)
    elif method == "GradCAM++":
        cam = GradCamPlusPlus(net, layer_name)
    return cam

def gen_cam(image_dir, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    image = cv2.imread(image_dir)
    # mask->heatmap
    num = mask.shape[0]
    for i in range(0,num):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[0]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)

        # merge heatmap to original image
        cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam, (heatmap).astype(np.uint8)

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def Semantic_heatmap(mask1,mask2,image_dir,save_path):
    '''
    Semantic heatmap
    '''
    # Read image
    image = cv2.imread(image_dir)
    image = cv2.resize(image,(224,224))
    mask = mask1 * mask2
    num = mask.shape[0]
    for i in range(0,num):
        mask_ = np.float32(cv2.applyColorMap(np.uint8(norm_image(mask[i])), cv2.COLORMAP_JET))
        cv2.imwrite(os.path.join(save_path,str(i)+'.jpg'),0.5*image+0.5*mask_)

def main(args):
    # Read the images as input
    images = dl.Image_precessing(args.image_input)
    inputs = torch.tensor([images], requires_grad=True)
    # Load the networks
    net1 = get_network(args.network1)
    net2 = get_network(None,args.network2)
    # Heatmap
    cam1 = get_heatmap_method(net1, method=args.heatmap_method)
    cam2 = get_heatmap_method(net2, method=args.heatmap_method)
    # Mask
    mask1 = cam1(inputs.cuda(),1)  # cam mask
    mask2 = cam2(inputs.cuda())  # cam mask
    cam1.remove_handlers()
    cam2.remove_handlers()
    # gen_cam(args.image_input,mask1)
    # gen_cam(args.image_input,mask2)
    # Semantic heatmap
    Semantic_heatmap(mask1,mask2,args.image_input,args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-input', type=str, default="/home/cry/data1/VGGFace2-pytorch/VGGFace2/train/n000002/0001_01.jpg",
                        help='input image path')
    parser.add_argument('--network1', type=str, default='VGGFace2',
                        help='Face identity recognition network.')
    parser.add_argument('--network2', type=str, default='./pre-trained/attribute.pth',
                        help='Attribute network, name or path.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    args = parser.parse_args()
    
    mkdir(args.output_dir)
    
    main(args)