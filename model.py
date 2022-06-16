import torch.nn as nn
import torch
import torch.nn.functional as F
from math import ceil
    
class img_feature(nn.Module):
    def __init__(self, model):
        super(img_feature, self).__init__()
        self.features = model

        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        
    def split_img(self, x, div_parts):
        n, c, w_, h_ = x.size()
        w, h = ceil(w_ / 32) *  32, ceil(h_ / 32) *  32
        pad_w = torch.zeros(n, c, w - w_, h_).cuda()
        pad_h = torch.zeros(n, c, w, h - h_).cuda()
        x = torch.cat((x, pad_w), 2)
        x = torch.cat((x, pad_h), 3)
        
        block_size = w // ceil(div_parts / 2)
        l = []
        for i in range(div_parts):
            for j in range(div_parts):
                l.append(x[:, :, int(i / 2 * block_size): int((i / 2 + 1) * block_size), int(j / 2 * block_size): int((j / 2 + 1) * block_size)].unsqueeze(1))
        x = torch.cat(l, 1).reshape(-1, c, block_size, block_size)
        
    def split_img2(self, x, div_parts):
        n, c, w_, h_ = x.size()
        w, h = ceil(w_ / 32) *  32, ceil(h_ / 32) *  32
#         pad_w = torch.zeros(n, c, w - w_, h_).cuda()
#         pad_h = torch.zeros(n, c, w, h - h_).cuda()
#         x = torch.cat((x, pad_w), 2)
#         x = torch.cat((x, pad_h), 3)
        
        block_size = w // div_parts
        l = []
        for i in range(div_parts):
            for j in range(div_parts):
                l.append(x[:, :, int(i * block_size): int((i + 1) * block_size), int(j * block_size): int((j + 1) * block_size)].unsqueeze(1))
        x = torch.cat(l, 1).reshape(-1, c, block_size, block_size)
#         print(x.shape)
        return x

    def forward(self, x):
        bs = x.size(0)
        l = []
        for i in range(1,6):
            x_ = self.split_img2(x, i)
            x_ = self.features(x_)
            x_ = self.avg(x_).reshape(bs, x_.size(0) // bs, -1)
            l.append(x_)
        x = torch.cat(l, 1)
    
        return x
    
class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class projector(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=2048, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, output_dim=1024):
        super(projector, self).__init__()
        self.img_fc = ImgNN(img_input_dim, img_output_dim)
        self.text_fc = TextNN(text_input_dim, text_output_dim)
        self.fc = nn.Linear(img_output_dim, output_dim)

    def forward(self, img, text):
        
        view1_feature = self.img_fc(img)
        view2_feature = self.text_fc(text)
        view1_feature = self.fc(view1_feature)
        view2_feature = self.fc(view2_feature)
        
        return view1_feature, view2_feature
      
class projector(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=2048, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=512, output_dim=2048):
        super(projector, self).__init__()
        self.text_fc = TextNN(text_input_dim, text_output_dim)
        self.fc = nn.Linear(text_output_dim, output_dim)

    def forward(self, text):
        
        view_feature = self.text_fc(text)
        view_feature = self.fc(view_feature)
        
        return view_feature
    
class classifier(nn.Module):
    def __init__(self, feature_size, classes_num):
        super(classifier, self).__init__()
        
        self.num_ftrs = 2048

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, classes_num),
#             nn.BatchNorm1d(classes_num),
        )

    def forward(self, f):
        
        x = self.classifier(f)
        
        return x
    
class prototype(nn.Module):
    def __init__(self, init_prototype):
        super(prototype, self).__init__()

        self.prototype = torch.nn.Parameter(init_prototype)

    def forward(self, f, targets):
        batch_size = f.size(0)
        prototype_pair1 = self.prototype.unsqueeze(0).repeat(batch_size, 1, 1)[torch.arange(0,batch_size), targets].unsqueeze(1)
        prototype_pair2 = self.prototype.unsqueeze(0).repeat(batch_size, 1, 1)[torch.arange(0,batch_size), targets.flip(dims=[0])].unsqueeze(1)
        prototype_pair = torch.cat((prototype_pair1, prototype_pair2), 1)
#         print(prototype_pair.shape, f.shape)
        bi_pred = torch.einsum("ijk,ik->ij", prototype_pair, f)
        bi_pred = torch.nn.functional.normalize(bi_pred, dim=-1)
        
        return bi_pred
    
class refine(nn.Module):
    def __init__(self, channel):
        super(refine, self).__init__()
        self.inter_channel = channel // 2
        self.phi = nn.Linear(channel, self.inter_channel)
        self.theta = nn.Linear(channel, self.inter_channel)
        self.g = nn.Linear(channel, self.inter_channel)
        self.attention = nn.Sequential(
#                          nn.Linear(self.inter_channel, self.inter_channel),
# #                          nn.BatchNorm2d((55, self.inter_channel)),
#                          nn.ReLU(inplace=True),
                         nn.Linear(self.inter_channel, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [N, C, H , W]
        b, p, c = x.size()
        # [N, C/2, H * W]
        x_phi = self.phi(x)
        # [N, H * W, C/2]
        x_theta = self.theta(x)
        # [N, H * W, H * W]
        x_g = self.g(x)
        mul_theta_phi = torch.einsum('ijm,ikm->ijk', x_phi, x_theta)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        m = torch.einsum('ijk,ikm->ijm', mul_theta_phi, x_g)
#         return self.softmax(mask + vf)
#         return self.softmax(mask)
        m = self.attention(m).squeeze(-1)
        m = torch.nn.functional.normalize(m, dim=-1)
        

#         return self.softmax(mask * 10)
        return m, self.softmax(m * 10)

class base_model(nn.Module):
    def __init__(self, model, classes_num):
        super(base_model, self).__init__()

        self.features = model
#         self.features = nn.Sequential(*list(model.children())[:-2])
        self.max3 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, classes_num),
        )

    def forward(self, x):
        xf = self.features(x)

        x = self.max3(xf)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        x = self.classifier(x)
        