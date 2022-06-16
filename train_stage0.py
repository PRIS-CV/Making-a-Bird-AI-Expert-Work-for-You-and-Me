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


def train0(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):

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
#         transforms.CenterCrop(224),
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
        img_net = load_model(model_name='img_feature', pretrain=True, require_grad=True)
        projector = load_model(model_name='projector')
        expert_att = load_model(model_name='refine')
        classifier1, classifier2, classifier3 = load_model(model_name='classifier')
        refine_net = load_model(model_name='refine')

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
    l = []
    for i in range(1,6):
        l.append(torch.Tensor([1 / i / i]).repeat(i * i))
    vf0 = torch.cat(l, 0).unsqueeze(0).unsqueeze(2).cuda() / 5
    
    
    ###################################################
    optimizer = optim.SGD([
        {'params': img_net.parameters(), 'lr': 0.001},
        {'params': classifier1.parameters(), 'lr': 0.01}

    ],
        momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.001)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.01)
#         for param_group in optimizer.param_groups:
#             if epoch == 50 or epoch == 75:
#                 param_group['lr'] = param_group['lr'] / 10
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            
#         continue
        img_net.train()
        projector.eval()
        classifier1.train()
        expert_att.train()
        refine_net.eval()
        train_loss = 0
        correct = 0
        retrival_correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, txt_features, targets) in enumerate(trainloader):
#             print(features)
#             print(torch.min(features), torch.mean(features), torch.max(features))
            idx = batch_idx
#             if idx == 1:
#                 break
            if use_cuda:
                inputs, targets, txt_features = inputs.cuda(), targets.cuda(), txt_features.cuda()
            inputs, targets, txt_features = Variable(inputs), Variable(targets), Variable(txt_features)
            
            optimizer.zero_grad()
            img_features = img_net(inputs)
            raw_outputs = classifier1(torch.sum(img_features * vf0, 1))
            
            loss = CELoss(raw_outputs, targets) * 1
            loss.backward()
            optimizer.step()
        torch.save(img_net.state_dict(), './' + store_name + '/img_net.pth')
        if idx % 100 == 0:
            print(loss)



