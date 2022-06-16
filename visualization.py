from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
from folder import *
from PIL import Image
from PIL import ImageDraw
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
from folder import *

def tensor2img(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = (input_tensor + 1) / 2
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    im = Image.fromarray(input_tensor)
    im.save(filename)
    print(filename)
    
def clamp(mmin, mmax, x):
    if x > mmax:
        x = mmax
    if x < mmin:
        x = mmin
    return x

def get_block_num(x):
    c = 0
    x = x + 1
    for i in range(10):
        if x <= c + (i + 1) ** 2:
            return i + 1, c
        c = c + (i + 1) ** 2
    
def bbox2img(input_tensor, bboxs, filename, idx):
    
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = (input_tensor + 1) / 2
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    
    output_tensor = input_tensor.copy() // 3 + 170
    for i in idx:
        block_num, cut = get_block_num(bboxs[i])
        w = int((bboxs[i] - cut) % block_num * 224 / block_num)
        h = int((bboxs[i] - cut) // block_num * 224 / block_num)
        output_tensor[h: clamp(0, 223, h + 224//block_num), w: clamp(0, 223, w + 224//block_num)] = input_tensor[h: clamp(0, 223, h + 224//block_num), w: clamp(0, 223, w + 224//block_num)]
    
    im = Image.fromarray(output_tensor)
    
    draw = ImageDraw.Draw(im) 
    for i in idx:
        color = [255,0,0]
#         color[i] += 255
        color = tuple(color)
        block_num, cut = get_block_num(bboxs[i])
        w = int((bboxs[i] - cut) % block_num * 224 / block_num)
        h = int((bboxs[i] - cut) // block_num * 224 / block_num)
        draw.rectangle([w, h, clamp(0, 223, w + 224//block_num), clamp(0, 223, h + 224//block_num)], outline=color)
    
    im.save(filename)
    print(filename)
    
def split_img(x, div_parts):
    n, c, w_, h_ = x.size()
    w, h = ceil(w_ / 32) *  32, ceil(h_ / 32) *  32
        
    block_size = w // div_parts
    l = []
    for i in range(div_parts):
        for j in range(div_parts):
            l.append(x[:, :, int(i * block_size): int((i + 1) * block_size), int(j * block_size): int((j + 1) * block_size)].unsqueeze(1))
    x = torch.cat(l, 1).reshape(-1, c, block_size, block_size)
#         print(x.shape)
    return x

def vis(img_net, projector, expert_att, refine_net, batch_size):
    img_net.eval()
    projector.eval()
    refine_net.eval()
    img_net.cuda()
    expert_att.cuda()
    projector.cuda()
    refine_net.cuda()
    use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0,1,2,3")
    device = torch.device('cuda')

    transform_test = transforms.Compose([
        transforms.Scale((300, 300)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = ImageFolder(root='../data/fg2/Birds2/test/', transform=transform_test, return_path=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)

    for batch_idx, (inputs, txt_features, targets, path) in enumerate(trainloader):
        print(path)

        idx = batch_idx
        
        if use_cuda:
            inputs, targets, txt_features = inputs.cuda(), targets.cuda(), txt_features.cuda()
        inputs, targets, txt_features = Variable(inputs), Variable(targets), Variable(txt_features)

        img_features = img_net(inputs)
        vf0_, vf0 = expert_att(img_features)
        raw_img_features = torch.sum(img_features * vf0.unsqueeze(-1), 1)

        txt_features = projector(txt_features) # bi-modal features alignment for retrival.
        vf_, vf = get_vf(img_features.detach(), txt_features)
        cap_img_features = torch.sum(img_features.detach() * vf.unsqueeze(-1), 1)

        un_vf = torch.clamp(vf0 - vf*2, min=0, max=1)
        un_vf = torch.nn.functional.normalize(un_vf, dim=-1)
        un_vf = F.softmax(un_vf * 10, 1)
        un_img_features = img_features.detach() * un_vf.unsqueeze(-1)

        corner_, corner = refine_net(un_img_features.detach())
        corner_vf = torch.nn.functional.normalize(corner * un_vf.detach(), dim=-1)
        corner_vf = F.softmax(corner_vf * 10, 1)

        for p in range(1,8):
            for i in range(200):
                if not os.path.exists("vis_output/part_{}/{}".format(p, path[0].split("/")[-2])):
                    os.makedirs("vis_output/part_{}/{}".format(p, path[0].split("/")[-2])) 
            import pdb
            pdb.set_trace()         
            _, id_re_vf = torch.topk(corner_vf.squeeze(0)[0:], p, dim = 0)
            bbox2img(inputs, id_re_vf, "vis_output/part_{}".format(p) + path[0].split('train')[-1], range(p))
        
    return None

img_net = load_model(model_name='img_feature', pretrain=True, require_grad=False)
projector = load_model(model_name='projector', require_grad=False)
expert_att = load_model(model_name='refine', require_grad=False)
refine_net = load_model(model_name='refine')

name = "model_dir"
img_net.load_state_dict(torch.load(name + '/img_net.pth'))
expert_att.load_state_dict(torch.load(name + '/expert_att.pth'))
projector.load_state_dict(torch.load(name + '/projector.pth'))
refine_net.load_state_dict(torch.load(name + '/refine_net.pth'))

vis(img_net, projector, expert_att, refine_net, 1)