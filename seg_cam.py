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

from utils import *
from models.BiSeNet.BiSeNet import BiSeNet


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    #save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.open(dspth)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # Detect which type? [0 1 17]
        print(np.unique(parsing))

        vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path="./results/test.jpg")

def main(args):
    evaluate(dspth=args.image_input, cp='./pre-trained/BiSeNet.pth')

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