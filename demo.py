# -*- coding: utf-8 -*-  

"""
Created on 2021/2/3

@author: Ruoyu Chen
"""

import argparse
import cv2

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network1', type=str, default='VGGFace2',
                        help='Face identity recognition network.')
    parser.add_argument('--network2', type=str, default='resnet50',
                        help='Attribute network.')
    