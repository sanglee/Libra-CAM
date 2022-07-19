#coding=utf-8
# Generating video using I-GOS
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)
from .IGOS_util_vgg import *
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


def Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224), Gaussian_param = [51, 50], Median_param = 11, blur_type= 'Gaussian', use_cuda = 1):
    ########################
    # Generate blurred images as the baseline

    # Parameters:
    # -------------
    # input_img: the original input image
    # img_label: the classification target that you want to visualize (img_label=-1 means the top 1 classification label)
    # model: the model that you want to visualize
    # resize_shape: the input size for the given model
    # Gaussian_param: parameters for Gaussian blur
    # Median_param: parameters for median blur
    # blur_type: Gaussian blur or median blur or mixed blur
    # use_cuda: use gpu (1) or not (0)
    ####################################################

#    original_img = cv2.imread(input_img, 1)
#    original_img = cv2.resize(original_img, resize_shape)

#    original_img = cv2.resize(input_img, resize_shape)
#    img = np.float32(original_img) #/ 255
    
    img = np.float32(input_img)
    
    if blur_type =='Gaussian':   # Gaussian blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

    elif blur_type == 'Median': # Median blur
        Kernelsize_M = Median_param
        blurred_img = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

    elif blur_type == 'Mixed': # Mixed blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img1 = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

        Kernelsize_M = Median_param
        blurred_img2 = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

        blurred_img = (blurred_img1 + blurred_img2) / 2

    return img, blurred_img


def Integrated_Mask(base_mask,img, blurred_img, model, category, max_iterations = 15, integ_iter = 20,
                    tv_beta=2, l1_coeff = 0.01*300, tv_coeff = 0.2*300, size_init = 112, use_cuda =1):
    ########################
    # IGOS: using integrated gradient descent to find the smallest and smoothest area that maximally decrease the
    # output of a deep model

    # Parameters:
    # -------------
    # img: the original input image
    # blurred_img: the baseline for the input image
    # model: the model that you want to visualize
    # category: the classification target that you want to visualize (category=-1 means the top 1 classification label)
    # max_iterations: the max iterations for the integrated gradient descent
    # integ_iter: how many points you want to use when computing the integrated gradients
    # tv_beta: which norm you want to use for the total variation term
    # l1_coeff: parameter for the L1 norm
    # tv_coeff: parameter for the total variation term
    # size_init: the resolution of the mask that you want to generate
    # use_cuda: use gpu (1) or not (0)
    ####################################################

    # preprocess the input image and the baseline image
    img = preprocess_image(img, use_cuda, require_grad=False)
    
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    resize_size = img.data.shape
    resize_wh = (img.data.shape[2], img.data.shape[3])

    if use_cuda:
        zero_img = Variable(torch.zeros(resize_size).cuda(), requires_grad=False)
    else:
        zero_img = Variable(torch.zeros(resize_size), requires_grad=False)


    ##########################################
    #For Sig CAM
    # initialize the mask
    mask_init = base_mask 
    mask = numpy_to_torch(mask_init, use_cuda, requires_grad=True)
    ##########################################

    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    # You can choose any optimizer
    # The optimizer doesn't matter, because we don't need optimizer.step(), we just use it to compute the gradient
    optimizer = torch.optim.Adam([mask], lr=0.1)
    #optimizer = torch.optim.SGD([mask], lr=0.1)

    target = torch.nn.Softmax(dim=1)(model(img))
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    # if category=-1, choose the original top 1 category as the one that you want to visualize
    if category ==-1:
        category = category_out

    '''
    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")
    '''

    curve1 = np.array([])
    curve2 = np.array([])
    curvetop = np.array([])


    # Integrated gradient descent
    alpha = 0.0001
    beta = 0.2

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channels
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                  upsampled_mask.size(3))


        # the l1 term and the total variation term
        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
                tv_coeff * tv_norm(mask, tv_beta)
        loss_all = loss1.clone()

        # compute the perturbed image
        perturbated_input_base = img.mul(upsampled_mask) + \
                                 blurred_img.mul(1 - upsampled_mask)


        for inte_i in range(integ_iter):


            # Use the mask to perturbated the input image.
            integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask


            perturbated_input_integ = img.mul(integ_mask) + \
                                     blurred_img.mul(1 - integ_mask)

            # add noise
            noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            noise = numpy_to_torch(noise, use_cuda, requires_grad=False)

            perturbated_input = perturbated_input_integ + noise

            new_image = perturbated_input
            outputs = torch.nn.Softmax(dim=1)(model(new_image))
            loss2 = outputs[0, category]

            loss_all = loss_all + loss2/20.0


        # compute the integrated gradients for the given target,
        # and compute the gradient for the l1 term and the total variation term
        optimizer.zero_grad()
        loss_all.backward()
        whole_grad = mask.grad.data.clone()

        loss2_ori = torch.nn.Softmax(dim=1)(model(perturbated_input_base))[0, category]



        loss_ori = loss1 + loss2_ori
        if i==0:
            if use_cuda:
                curve1 = np.append(curve1, loss1.data.cpu().numpy())
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
                curvetop = np.append(curvetop, loss2_ori.data.cpu().numpy())

            else:
                curve1 = np.append(curve1, loss1.data.numpy())
                curve2 = np.append(curve2, loss2_ori.data.numpy())
                curvetop = np.append(curvetop, loss2_ori.data.numpy())



        if use_cuda:
            loss_oridata = loss_ori.data.cpu().numpy()
        else:
            loss_oridata = loss_ori.data.numpy()



        # LINE SEARCH with revised Armijo condition
        step = 200.0
        MaskClone = mask.data.clone()
        MaskClone -= step * whole_grad
        MaskClone = Variable(MaskClone, requires_grad=False)
        MaskClone.data.clamp_(0, 1) # clamp the value of mask in [0,1]


        mask_LS = upsample(MaskClone)   # Here the direction is the whole_grad
        Img_LS = img.mul(mask_LS) + \
                 blurred_img.mul(1 - mask_LS)
        outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
        loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + \
                  tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

        if use_cuda:
            loss_LSdata = loss_LS.data.cpu().numpy()
        else:
            loss_LSdata = loss_LS.data.numpy()


        new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        new_condition = new_condition.sum()
        new_condition = alpha * step * new_condition

        while loss_LSdata > loss_oridata - new_condition.cpu().numpy():
            step *= beta

            MaskClone = mask.data.clone()
            MaskClone -= step * whole_grad
            MaskClone = Variable(MaskClone, requires_grad=False)
            MaskClone.data.clamp_(0, 1)
            mask_LS = upsample(MaskClone)
            Img_LS = img.mul(mask_LS) + \
                     blurred_img.mul(1 - mask_LS)
            outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
            loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + \
                      tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

            if use_cuda:
                loss_LSdata = loss_LS.data.cpu().numpy()
            else:
                loss_LSdata = loss_LS.data.numpy()


            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition

            if step<0.00001:
                break

        mask.data -= step * whole_grad

        #######################################################################################################


        if use_cuda:
            curve1 = np.append(curve1, loss1.data.cpu().numpy())
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
        else:
            curve1 = np.append(curve1, loss1.data.numpy())
            curve2 = np.append(curve2, loss2_ori.data.numpy())

        mask.data.clamp_(0, 1)
        if use_cuda:
            maskdata = mask.data.cpu().numpy()
        else:
            maskdata = mask.data.numpy()

        maskdata = np.squeeze(maskdata)
        maskdata, imgratio = topmaxPixel(maskdata, 40)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)

        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)
        outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
        loss_top1 = l1_coeff * torch.mean(torch.abs(1 - Masktop)) + \
                    tv_coeff * tv_norm(Masktop, tv_beta)
        loss_top2 = outputstopLS[0, category]

        if use_cuda:
            curvetop = np.append(curvetop, loss_top2.data.cpu().numpy())
        else:
            curvetop = np.append(curvetop, loss_top2.data.numpy())


        if max_iterations >3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
#                    print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
#                    print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
                    l1_coeff = l1_coeff / 5


            #######################################################################################

    upsampled_mask = upsample(mask)

    if use_cuda:
        mask = mask.data.cpu().numpy().copy()
    else:
        mask = mask.data.numpy().copy()

    return mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category