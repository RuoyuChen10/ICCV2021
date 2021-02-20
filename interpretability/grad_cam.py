# -*- coding: utf-8 -*-
"""
Created on 2021/2/3

@author: Ruoyu Chen

"""
import numpy as np
import cv2
import torch


class GradCAM(object):
    """
    GradCAM
    """
    def __init__(self, net, layer_name):
        self._net_init(net)
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.task_num = self.get_multi_task_number(self.net)

    def _net_init(self,net):
        self.net = net
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple, length 1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def get_multi_task_number(self, net):
        """
        Get the number of multi-task
        The multi-task net must using torch.nn.Linear as judge,
        and only exist 1 layer for one task.
        """
        num = 0
        for name, m in net.named_modules():
            if isinstance(m, torch.nn.Linear):
                num+=1
        return num

    def __call__(self, inputs, index=None):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.handlers = []
        self._register_hook()
        self.net.zero_grad()

        total_cam = []
        output = self.net(inputs)  # [num,num_classes] -> _get_features_hook
        for i in range(self.task_num):
            # choose the task
            if self.task_num == 1:
                task_output = output
            else:
                task_output = output[i]
            if index is None:
                index = torch.argmax(task_output)
            print(index)
            target = task_output[0][index]
            if i == self.task_num-1:
                target.backward()  # -> _get_grads_hook
            else:
                target.backward(retain_graph=True)  # -> _get_grads_hook
            gradient = self.gradient[0]  # [C,H,W]
            weight = torch.mean(gradient, axis=(1, 2))  # [C]

            feature = self.feature[0]  # [C,H,W]

            cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
            cam = torch.sum(cam, axis=0)  # [H,W]
            cam = torch.relu(cam)  # ReLU

            # Normalization
            cam -= torch.min(cam)
            cam /= torch.max(cam)
            # resize to 224*224
            cam = cv2.resize(cam.cpu().data.numpy(), (224, 224))
            total_cam.append(cam)
        return np.array(total_cam)


class GradCamPlusPlus(GradCAM):
    '''
    GradCam++
    '''
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.handlers = []
        self._register_hook()
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # Normalization
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam
