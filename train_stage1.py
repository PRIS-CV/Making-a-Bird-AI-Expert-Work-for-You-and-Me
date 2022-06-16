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


def train1(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):

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
        projector = load_model(model_name='projector')
        expert_att = load_model(model_name='refine')
        classifier1, classifier2, classifier3 = load_model(model_name='classifier')
        refine_net = load_model(model_name='refine')
    img_net.load_state_dict(torch.load(store_name + '/img_net.pth'))
#     classifier1.load_state_dict(torch.load(store_name + '/classifier.pth'))

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
    
    
    ###################################################
    optimizer = optim.SGD([
        {'params': classifier1.parameters(), 'lr': 0.01},
        {'params': expert_att.parameters(), 'lr': 0.01}

    ],
        momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.01)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.01)
#         for param_group in optimizer.param_groups:
#             if epoch == 50 or epoch == 75:
#                 param_group['lr'] = param_group['lr'] / 10
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            
#         continue
        img_net.eval()
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
            if use_cuda:
                inputs, targets, txt_features = inputs.cuda(), targets.cuda(), txt_features.cuda()
            inputs, targets, txt_features = Variable(inputs), Variable(targets), Variable(txt_features)
            
            optimizer.zero_grad()
            img_features = img_net(inputs)
            vf0_, vf0 = expert_att(img_features)
            raw_outputs = classifier1(torch.sum(img_features * vf0.unsqueeze(-1), 1))
            
            loss = CELoss(raw_outputs, targets) * 1
            loss.backward()
            optimizer.step()

        if True:
            raw_acc, cap_acc, corner_acc, re_acc, retriv_acc1, retriv_acc2, retriv_acc3, retriv_acc4 = test(img_net, projector, expert_att, [classifier1, classifier2, classifier3], refine_net, CELoss, batch_size)
            if True:
                max_val_acc = raw_acc
                torch.save(expert_att.state_dict(), './' + store_name + '/expert_att.pth')
                torch.save(classifier1.state_dict(), './' + store_name + '/classifier.pth')
            with open(exp_dir + '/results_test1.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc = %.5f, test_acc = %.5f\n, test_acc = %.5f\n, retriv_acc1 = %.5f\n, retriv_acc2 = %.5f\n, retriv_acc3 = %.5f\n, retriv_acc4 = %.5f\n' % (epoch, raw_acc, cap_acc, corner_acc, re_acc, retriv_acc1, retriv_acc2, retriv_acc3, retriv_acc4))



