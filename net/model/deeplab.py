# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

class VGG(nn.Module):

    def make_layers(self, in_shape, cfg, batch_norm):
        in_channels, height, width = in_shape

        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def __init__(self, in_shape, num_classes, cfg, batch_norm=False):
        super(VGG, self).__init__()
        in_channels, height, width = in_shape

        self.features = self.make_layers(cfg=cfg, in_shape=in_shape, batch_norm=batch_norm)
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = x
        prob = F.sigmoid(logit)
        return logit, prob

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class VGG_deeplab(nn.Module):

    def make_layers(self, in_shape):
        in_channels, height, width = in_shape

        layers = []

        # Conv1
        conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Conv2
        conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Conv3
        conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Conv4
        conv2d = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]

        # Conv5
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=2,  dilation=2)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=2,  dilation=2)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=2,  dilation=2)
        layers += [conv2d, nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]

        return nn.Sequential(*layers)


    def __init__(self, in_shape, num_classes):
        super(VGG_deeplab, self).__init__()
        in_channels, height, width = in_shape

        self.features = self.make_layers(in_shape=in_shape)
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=6, dilation=6), # fc6
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0),  # fc final
        )

        self.fc6_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=12, dilation=12),  # fc6
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0),  # fc final
        )

        self.fc6_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=18, dilation=18),  # fc6
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0),  # fc final
        )

        self.fc6_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=24, dilation=24),  # fc6
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0),  # fc final
        )


    def forward(self, x):
        x = self.features(x)
        # out1 = self.fc6_1(x)
        # out2 = self.fc6_2(x)
        # out3 = self.fc6_3(x)
        # out4 = self.fc6_4(x)
        # out = out1 + out2 + out3 + out4

        out = self.fc6_1(x)
        return out



########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1 #1
    num_classes = 1
    C,H,W =  3,640,640  #3,256,256
    inputs   = torch.randn(batch_size,C,H,W)
    labels = torch.FloatTensor(batch_size, H,W).random_(1)
    in_shape = inputs.size()[1:]

    if 1:
        net = VGG_deeplab(in_shape=in_shape, num_classes=num_classes).cuda().train()

        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        logits = net.forward(x)
        logits = F.upsample_bilinear(logits, (H, W))
        loss = BCELoss2d()(logits, y)
        loss.backward()

        pass
        #input('Press ENTER to continue.')

