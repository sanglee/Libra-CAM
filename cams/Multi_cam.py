import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import os
from cams.relevance_cam_utils.LRP_util import *

import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import os
from cams.AGF_utils.utils import *

def agf(test_model,loaded_model,inputs,**kwargs):
    
    model = loaded_model

    model.eval()
    
    if test_model =='vgg':
        layer = 'layer43'
        
    elif test_model == 'resnet':
        layer = 'layer4'

    
    temp = inputs[0].detach()
    in_tensor = inputs.cuda()
    output = model(in_tensor)
    #######################################################################################
    #SigCAM
    AGF = model.AGF(**kwargs)[0, 0].data.cpu().numpy()
    
    cam = AGF #AGF
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    #######################################################################################
    return cam

def multi_CAM(test_model,loaded_model,inputs):

    model = loaded_model
    
    model.eval()
    
    if test_model =='vgg':
        arg_target_layer = 43
        target_layer = model.features[int(arg_target_layer)]
        layer_path = int(arg_target_layer)
    elif test_model == 'resnet':
        arg_target_layer = 4
        target_layer = model.layer4
        layer_path = 'layer4'
    
    CAM_CLASS = GradCAM_multi(model, target_layer)
    Score_CAM_class = ScoreCAM(model,target_layer)
    
    in_tensor = inputs.cuda()
    output = model(in_tensor)
    maxindex = np.argmax(output.data.cpu().numpy())
#    maxindex = label
    
    Tt, Tn = CLRP(output, maxindex)
    posi_R = model.relprop(Tt,1,flag=layer_path).data.cpu().numpy()
    nega_R = model.relprop(Tn,1,flag=layer_path).data.cpu().numpy()
    
    R = posi_R - nega_R
    R = np.transpose(R[0],(1,2,0))
    r_weight = np.sum(R,axis=(0,1),keepdims=True)
    activation, grad_cam, grad_campp = CAM_CLASS(in_tensor, class_idx=maxindex)
    
    score_map, _ = Score_CAM_class(in_tensor, class_idx=maxindex)
    score_map = score_map.squeeze()
    score_map = score_map.detach().cpu().numpy()

    R_cam = np.sum(activation * r_weight, axis=-1)
    
    # 0 ~ 1 normalization
    R_cam = R_cam - np.min(R_cam)
    R_cam = R_cam / np.max(R_cam)
    R_cam = cv2.resize(R_cam,(224,224))
    
    #normalization
    grad_cam = grad_cam - np.min(grad_cam)
    grad_cam = grad_cam / np.max(grad_cam)

    grad_campp = grad_campp - np.min(grad_campp)
    grad_campp = grad_campp / np.max(grad_campp)
    
    score_map = score_map - np.min(score_map)
    score_map = score_map / np.max(score_map)
    
    return grad_cam,grad_campp,R_cam,score_map