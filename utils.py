# -*- coding: utf-8 -*-  

"""
Created on 2021/2/3

@author: Ruoyu Chen
"""

import os
import torch
import torch.nn as nn

def get_network(command,weight_path=None):
    '''
    Get the object network
        command: Type of network
        weight_path: If need priority load the pretrained model?
    '''
    # Load model
    if weight_path is not None and os.path.exists(weight_path):
        model = torch.load(weight_path)
        try:
            # if multi-gpu model:
            model = model.module
        except:
            # just 1 gpu or cpu
            pass
        pretrain = model.state_dict()
        new_state_dict = {}
        for k,v in pretrain.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
        print("Model parameters: " + weight_path + " has been load!")
        return model
    elif command == "resnet50":
        from models.resnet import resnet50
        print("Model load: ResNet50 as backbone.")
        return resnet50()
    elif command == 'VGGFace2':
        from models.vggface_models.resnet import resnet50
        weight_path = "./pre-trained/resnet50_scratch_weight.pkl"
        net = resnet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        net.load_state_dict(weights, strict=True)
        return net
    

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0