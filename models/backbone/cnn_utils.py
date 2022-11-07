#encoding:utf-8
import torchvision, torch, sys, time, math
from torchvision import datasets, models, transforms

import torch.nn as nn
sys.path.insert(0, sys.path[0] + '/models/backbones')
from models.backbone.conveXt import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from models.backbone.pplcnet import PPLCNet_x0_25, PPLCNet_x0_35, PPLCNet_x0_5, PPLCNet_x0_75, PPLCNet_x1_0, PPLCNet_x1_5, PPLCNet_x2_0, PPLCNet_x2_5
from models.backbone.origin_pplcnet import PPLCNet_x1_0 as origin_pplcx10

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, class_num=3):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    # input_size = 0

    if model_name == 'conveXt_tiny_48x96x192x384':
        model_ft = convnext_tiny(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], class_num=class_num)
    elif model_name == 'conveXt_tiny_96x192x384x768':
        model_ft = convnext_tiny(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], class_num=class_num)
    elif model_name == 'pplcx10':
        model_ft = PPLCNet_x1_0(class_num=class_num)
    elif model_name == 'pplcx075':
        model_ft = PPLCNet_x0_75(class_num=class_num)    
    elif model_name == 'pplcx05':
        model_ft = PPLCNet_x0_5(class_num=class_num)   
    elif model_name == 'pplcx035':
        model_ft = PPLCNet_x0_35(class_num=class_num)   
    elif model_name == 'pplcx025':
        model_ft = PPLCNet_x0_25(class_num=class_num)   
    elif model_name == 'origin_pplcx10':
        model_ft = origin_pplcx10(class_num=class_num)
    


    # elif model_name == 'pplcnetx025':
    #     model_ft = PPLCNet_x0_25(pretrained='.pretrained/PPLCNet_x0_25_pretrained.pth', class_num=num_classes)
        
    # elif model_name == 'pplcnetx035':
    #     model_ft = PPLCNet_x0_35(pretrained='.pretrained/PPLCNet_x0_35_pretrained.pth', class_num=num_classes)  

    # elif model_name == 'pplcnetx05':
    #     model_ft = PPLCNet_x0_5(pretrained='.pretrained/PPLCNet_x0_5_ssld_pretrained.pth', class_num=num_classes) 

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft

def load_model(model_path):
    model_ft = torch.load(model_path)
    # set_parameter_requires_grad(model_ft, feature_extract)
    # model_ft.classifier = model_ft.classifier # nn.Linear(num_ftrs, num_classes) 
    # input_size = 224
    return model_ft

def load_eval_model(backbone_type, model_path, num_classes, device):
    model = initialize_model(backbone_type, num_classes, False)
    
    # model = nn.DataParallel(model.to(device), device_ids=[0])
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def weight_DP_to_single(backbone_type, class_num, in_weight_file, out_weight_file):
    model = initialize_model(backbone_type, class_num)
    model = nn.DataParallel(model.to('cuda:0'), device_ids=[0])
    checkpoint = torch.load(in_weight_file)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    torch.save(model.module.state_dict(), out_weight_file)



if __name__ == "__main__":
    # backbone = initialize_model('se-resnext50', 10, False)
    
    # summary(backbone, torch.zeros((1, 3, 224, 224)))
    backbone_type = 'mobilenet'
    in_weight_file = 'outputs/best_models/DP_mobilenet_best.pth'
    out_weight_file = 'outputs/best_models/mobilenet_best.pth'
    weight_DP_to_single(backbone_type, in_weight_file, out_weight_file)
