# -*- coding: utf-8 -*-  

"""
Created on 2021/2/3

@author: Ruoyu Chen
"""

def get_network(command,weight_path=None):
    '''
    Get the object network
        command: Type of network
        weight_path: If need priority load the pretrained model?
    '''
    # Load model
    if weight_path is not None and os.path.exists(weight_path):
        model = torch.load(weight_path)
        print("Model parameters: " + weight_path + " has been load!")
    elif command == "resnet50":
        model = resnet50()
        print("Model load: ResNet50 as backbone.")
    # Multi GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model