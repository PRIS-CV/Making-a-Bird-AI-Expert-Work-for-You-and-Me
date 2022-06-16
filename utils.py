import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
from folder import *

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)

def load_model(model_name, pretrain=True, require_grad=True, init=None):
    print('==> Building model..')
    if model_name == 'projector':
        net = projector()
    if model_name == 'classifier':
        net = classifier(512, 200)
    if model_name == 'refine':
        net = refine(2048)
    if model_name == 'prototype':
        net = prototype(init)
    if model_name == 'img_feature':
        net = resnet50(pretrained=pretrain)
        net = img_feature(net)
    if model_name == 'classifier':
        net = [net, net, net]
    if model_name == 'resnet50':
        net = resnet50(pretrained=pretrain)
        net = base_model(net, 13)
    return net

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def test(img_net, projector, expert_att, classifier, refine_net, criterion, batch_size):
    classifier1, classifier2, classifier3 = classifier
    img_net.eval()
    projector.eval()
    expert_att.eval()
    classifier1.eval()
    classifier2.eval()
    classifier3.eval()
    refine_net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    raw_correct = 0
    cap_correct = 0
    corner_correct = 0
    re_correct = 0
    total = 0
    idx = 0
#     device = torch.device("cuda:0,1,2,3")
    device = torch.device('cuda')
   

    transform_test = transforms.Compose([
        transforms.Scale((300)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = ImageFolder(root='../data/fg2/Birds2/test/', transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16)

    for batch_idx, (inputs, txt_features, targets) in enumerate(trainloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets, txt_features = inputs.cuda(), targets.cuda(), txt_features.cuda()
        inputs, targets, txt_features = Variable(inputs), Variable(targets), Variable(txt_features)
        
        if True:
            img_features = img_net(inputs)
            vf0_, vf0 = expert_att(img_features)
            raw_outputs = classifier1(torch.sum(img_features * vf0.unsqueeze(-1), 1))
            raw_img_features = torch.sum(img_features * vf0.unsqueeze(-1), 1)
            
            txt_features = projector(txt_features)
            vf_, vf = get_vf(img_features.detach(), txt_features)
            cap_img_features = torch.sum(img_features.detach() * vf.unsqueeze(-1), 1)
            cap_outputs = classifier1(cap_img_features) # The second output
            
            un_vf = torch.clamp(vf0 - vf, min=0, max=1)
            un_vf = torch.nn.functional.normalize(un_vf, dim=-1)
            un_vf = F.softmax(un_vf * 10, 1)
            un_img_features = img_features.detach() * un_vf.unsqueeze(-1)
            
            corner_, corner = refine_net(un_img_features.detach())
            corner_vf = torch.nn.functional.normalize(corner * un_vf.detach(), dim=-1)
            corner_vf = F.softmax(corner_vf * 10, 1)
            corner_img_features = torch.sum(img_features.detach() * corner_vf.unsqueeze(-1), 1)
            corner_outputs = classifier1(corner_img_features)
            
            re_img_features  = (corner_img_features + cap_img_features.detach()) / 2
            re_outputs = classifier1(re_img_features)

        loss = criterion(re_outputs, targets)

        test_loss += loss.item()
        _, raw_predicted = torch.max(raw_outputs.data, 1)
        _, cap_predicted = torch.max(cap_outputs.data, 1)
        _, corner_predicted = torch.max(corner_outputs.data, 1)
        _, re_predicted = torch.max(re_outputs.data, 1)
        total += targets.size(0)
        raw_correct += raw_predicted.eq(targets.data).cpu().sum()
        cap_correct += cap_predicted.eq(targets.data).cpu().sum()
        corner_correct += corner_predicted.eq(targets.data).cpu().sum()
        re_correct += re_predicted.eq(targets.data).cpu().sum()

    raw_acc = 100. * float(raw_correct) / total
    cap_acc = 100. * float(cap_correct) / total
    corner_acc = 100. * float(corner_correct) / total
    re_acc = 100. * float(re_correct) / total
    test_loss = test_loss / (idx + 1)

    return raw_acc, cap_acc, corner_acc, re_acc

def get_vf(img_f, txt_f):
    img_f = torch.nn.functional.normalize(img_f, dim=-1)
    txt_f = torch.nn.functional.normalize(txt_f, dim=-1)
    vf_ = torch.einsum('ijk,ik->ij', img_f, txt_f)
    vf_ = torch.nn.functional.normalize(vf_, dim=-1)
    vf = F.softmax(vf_ * 10, 1)
    
    return vf_, vf

def contrastive_loss(cap_img_f, txt_f):
    cap_img_f = torch.nn.functional.normalize(cap_img_f, dim=-1)
    txt_f = torch.nn.functional.normalize(txt_f, dim=-1)
    CELoss = nn.CrossEntropyLoss()
    m = torch.einsum('im,jm->ij', txt_f, cap_img_f)
    label = torch.arange(0, m.size(0)).cuda()
    
    _, predicted = torch.max(m.data, 1)
    correct = predicted.eq(label.data).cpu().sum()
    
    return CELoss(m, label), correct