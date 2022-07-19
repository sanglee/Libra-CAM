import cv2
import torch
import numpy as np
from torch.nn import functional as F

def thresholding_mask(cam,threshold):
    cam_sorted_index = cam.reshape(-1).argsort()
    cam_mask_percent_index = int(len(cam_sorted_index) * (0.1 * threshold))
    masking_index = cam_sorted_index[:cam_mask_percent_index]
    cam1_binarize = cam.copy().reshape(-1)
    cam1_binarize[masking_index] = 0
    cam1_binarize = cam1_binarize.reshape((224,224))
    return cam1_binarize


def get_best_threshold(cam,model,orginal_softmax,label,inputs):
    
    original_confi = orginal_softmax.T[label][0].detach().cpu().numpy()        
    sorted_index = cam.reshape(-1).argsort()
    
    increase_percent_list = []
    for i in range(0,10):
       ##########################################
        mask_percent_index = int(len(sorted_index) * (0.1 * i))

        masking_index = sorted_index[:mask_percent_index]
        
        cam1_bandpass = cam.copy().reshape(-1)
        cam1_bandpass[masking_index] = 0
        cam1_bandpass = cam1_bandpass.reshape((224,224))
        
        marker1_band_pass = torch.tensor(cam1_bandpass)   
        ##########################################

        temp = inputs.detach().cpu()
        img = marker1_band_pass * temp
        
        del cam1_bandpass
        logits = model(img.cuda())

        masked_confi = F.softmax(logits).T[label][0].detach().cpu().numpy()

        avgincrease =  masked_confi - original_confi
        avgincrease = np.maximum(avgincrease,0)
        increase_percent_list.append(avgincrease)
        
    optimal_threshold = np.array(increase_percent_list).argmax() + 1
    cam = thresholding_mask(cam,optimal_threshold)
    return cam