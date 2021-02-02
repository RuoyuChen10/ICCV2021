# -*- coding: utf-8 -*-  

"""
Created on 2021/2/3

@author: Ruoyu Chen
"""

import argparse
import cv2

import Datasets.dataload as dl
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from utils import *

def main(args):
    images = Image_precessing(args.image_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-input', type=str, default=None,
                        help='input image path')
    parser.add_argument('--network1', type=str, default='VGGFace2',
                        help='Face identity recognition network.')
    parser.add_argument('--network2', type=str, default='resnet50',
                        help='Attribute network.')
    args = parser.parse_args()
    
    main(args)