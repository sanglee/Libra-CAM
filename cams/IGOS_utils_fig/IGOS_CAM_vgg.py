#coding=utf-8
# Generating video using I-GOS
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)
from .IGOS_util_vgg import *
from .IGOS_vgg import *
import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6" #그외 GPU는 삼성 SDS 사용  #, 6, 7"

def get_igos(base_mask,input_img,img_label,model):
        img, blurred_img = Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224),
                                                     Gaussian_param=[51, 50],
                                                     Median_param=11, blur_type='Gaussian', use_cuda=use_cuda)

        mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category = Integrated_Mask(base_mask,img, blurred_img, model,
                                                                                                 img_label,
                                                                                                 max_iterations=15,
                                                                                                 integ_iter=20,
                                                                                                 tv_beta=2,
                                                                                                 l1_coeff=0.01 * 100,
                                                                                                 tv_coeff=0.2 * 100,
                                                                                                 size_init=28,
                                                                                                 use_cuda=use_cuda)  #

        mask = upsampled_mask.detach().cpu().data.numpy()[0]
        mask = np.transpose(mask, (1,2,0))
        mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
        mask = 1 - mask
#        heatmap, IGOS, blurred = get_heatmap(upsampled_mask, img * 255, blurred_img, blur_mask=0)
        return mask.squeeze()