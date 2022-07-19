import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def Libra_cam(test_model,loaded_model,target_input,ref_libra,label,ref_cnt,org_target_class_proba):
    forward_result = []
    def forward_hook(module,finput, output):
        forward_result.append(torch.squeeze(output))
    
    backward_result = []
    def backward_hook(module,grad_input,grad_output):
        backward_result.append(torch.squeeze(grad_output[0]))
    
    
    model = loaded_model
    model.eval()
    
    target_layers_names = 'layer4'
    if test_model == 'vgg':
        target_layers_names = 'features'
        forward_handle = model._modules.get(target_layers_names)[-1].register_forward_hook(forward_hook)
        backward_handle = model._modules.get(target_layers_names)[-1].register_backward_hook(backward_hook)
    elif test_model == 'resnet':
        forward_handle = model._modules.get(target_layers_names).register_forward_hook(forward_hook)
        backward_handle = model._modules.get(target_layers_names).register_backward_hook(backward_hook)

    image_tensor = target_input
    image_tensor = image_tensor.cuda() 
    logits = model(image_tensor)
    score = logits.max()
    model.zero_grad()
    score.backward() 


    relu = torch.nn.ReLU()
    target_activation = forward_result[0]
    target_grad = backward_result[0]
    
    size_upsample = (224, 224) #setting image size manually -> ImageNet
    output_cam = []
    for i in range(0,ref_cnt):
        ref_activation = ref_libra[i]['activation']
        coeff = target_activation - torch.tensor(ref_activation).cuda() #base
        cam0 = torch.sum(target_grad * coeff, dim = 0).detach().cpu().numpy()
        #normalization
        cam0 = cam0 - np.min(cam0)
        cam_img0 = cam0 / np.max(cam0)
        output_cam.append(cv2.resize(cam_img0, size_upsample))

    forward_result.clear()
    backward_result.clear()
    forward_handle.remove()
    backward_handle.remove()
    

    ####################################
    #rejection criterion
    filter_output_cam = torch.tensor(output_cam).unsqueeze(dim=1).cuda()
    candidates = filter_output_cam * image_tensor
    temp_logits = model(candidates)
    temp_proba = F.softmax(temp_logits)
    ref_target_class_proba = temp_proba[:,label]
    
    candidate_avgincrease =  ref_target_class_proba - org_target_class_proba #[30,1]
    
    l_cam = np.array(output_cam) #27 224, 224    

    ######################################################
    #Add (m, sigma)
    relu_candidate_avgincrease = relu(candidate_avgincrease)
    sampled_idx = (relu_candidate_avgincrease > 0).squeeze().detach().cpu().numpy() #sampled idx
    if sampled_idx.sum() != 0:
        relu_candidate_avgincrease = relu_candidate_avgincrease.squeeze().detach().cpu().numpy() #(30,)
        l_cam = l_cam[sampled_idx,:,:]
        relu_candidate_avgincrease = relu_candidate_avgincrease[sampled_idx]
        m = relu_candidate_avgincrease.mean()
        std = np.std(relu_candidate_avgincrease)
        
        threshold_filter = m - std
        filterd_idx = (relu_candidate_avgincrease >= threshold_filter)
        relu_candidate_avgincrease = relu_candidate_avgincrease[filterd_idx]
        l_cam = l_cam[filterd_idx]
    else:
        avgincrease = candidate_avgincrease.squeeze().detach().cpu().numpy()
        m = avgincrease.mean()
        std = np.std(avgincrease)
        threshold_filter = m + std
        
        filterd_idx = (avgincrease > threshold_filter)
        if filterd_idx.sum() !=0:
            l_cam = l_cam[filterd_idx]
        #candidate_avgincrease

    l_cam = np.mean(l_cam,axis=0)
        
    del model, forward_result, backward_result
    
    return l_cam