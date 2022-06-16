from __future__ import print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import *
from folder import *


def train3(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):

    # setup output
    exp_dir = store_name

    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((300)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = ImageFolder(root='../data/fg2/Birds2/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

    # Model
    if resume:
        img_net = torch.load(model_path)
    else:
        img_net = load_model(model_name='img_feature', pretrain=True, require_grad=False)
        projector = load_model(model_name='projector', require_grad=False)
        expert_att = load_model(model_name='refine', require_grad=False)
        classifier1, classifier2, classifier3 = load_model(model_name='classifier', require_grad=False)
        refine_net = load_model(model_name='refine')
        
    img_net.load_state_dict(torch.load(store_name + '/img_net.pth'))
    expert_att.load_state_dict(torch.load(store_name + '/expert_att.pth'))
    classifier1.load_state_dict(torch.load(store_name + '/classifier.pth'))
    projector.load_state_dict(torch.load(store_name + '/projector.pth'))

    if use_cuda:
        img_net.cuda()
        projector.cuda()
        expert_att.cuda()
        classifier1.cuda()
        classifier2.cuda()
        classifier3.cuda()
        refine_net.cuda()
#         cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    KLDiv = nn.KLDivLoss()
#     l = []
#     for i in range(1,6):
#         l.append(torch.Tensor([1 / i / i]).repeat(i * i))
#     vf0 = torch.cat(l, 0).unsqueeze(0).unsqueeze(2).cuda() / 5
    
    
    ###################################################
    optimizer = optim.SGD([
        {'params': refine_net.parameters(), 'lr': 0.001}

    ],
        momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.01)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            
#         continue
        img_net.eval()
        projector.eval()
        expert_att.eval()
        classifier1.eval()
        refine_net.train()
        train_loss = 0
        correct = 0
        retrival_correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, txt_features, targets) in enumerate(trainloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets, txt_features = inputs.cuda(), targets.cuda(), txt_features.cuda()
            inputs, targets, txt_features = Variable(inputs), Variable(targets), Variable(txt_features)
            
            optimizer.zero_grad()
            img_features = img_net(inputs)
            vf0_, vf0 = expert_att(img_features)
            raw_outputs = classifier1(torch.sum(img_features * vf0.unsqueeze(-1), 1))
            
            txt_features = projector(txt_features) # bi-modal features alignment for retrival.
            vf_, vf = get_vf(img_features.detach(), txt_features)
            cap_img_features = torch.sum(img_features.detach() * vf.unsqueeze(-1), 1)
            
            un_vf = torch.clamp(vf0 - vf, min=0, max=1)
            un_vf = torch.nn.functional.normalize(un_vf, dim=-1)
            un_vf = F.softmax(un_vf * 10, 1)
            un_img_features = img_features.detach() * un_vf.unsqueeze(-1)
            
            corner_, corner = refine_net(un_img_features.detach())
            corner_vf = torch.nn.functional.normalize(corner * un_vf.detach(), dim=-1)
            corner_vf = F.softmax(corner_vf * 10, 1)
            corner_img_features = torch.sum(img_features.detach() * corner_vf.unsqueeze(-1), 1)
            
            re_img_features  = (corner_img_features + cap_img_features) / 2
            re_outputs = classifier1(re_img_features)
            
            T = 5
            loss = KLDiv(F.log_softmax(raw_outputs.detach() / T, -1), F.softmax(re_outputs / T, -1)) * 1
            loss.backward()
            optimizer.step()
            del txt_features, cap_img_features, img_features, un_img_features, corner_img_features, re_img_features, raw_outputs, re_outputs, vf_, vf, un_vf, corner_, corner

        if True:
            raw_acc, cap_acc, corner_acc, re_acc, retriv_acc1, retriv_acc2, retriv_acc3, retriv_acc4 = test(img_net, projector, expert_att, [classifier1, classifier2, classifier3], refine_net, CELoss, batch_size)
            if True:
                max_val_acc = re_acc
                torch.save(refine_net.state_dict(), './' + store_name + '/refine_net.pth')
            with open(exp_dir + '/results_test3.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc = %.5f, test_acc = %.5f\n, test_acc = %.5f\n, retriv_acc1 = %.5f\n, retriv_acc2 = %.5f\n, retriv_acc3 = %.5f\n, retriv_acc4 = %.5f\n' % (epoch, raw_acc, cap_acc, corner_acc, re_acc, retriv_acc1, retriv_acc2, retriv_acc3, retriv_acc4))




