import itertools

from torch import nn as nn
from torchvision.models import resnet50, alexnet
from torchvision.models.resnet import Bottleneck

import torch.nn.functional as F
from caffenet.caffenet_pytorch import load_caffenet
from models.model import BasicNet
from models.torch_future import Flatten
from models.autodial import MSAutoDIAL
from models.alexnet import AlexNetCaffe
import torch

class ResNet50(BasicNet):
    def __init__(self, domain_classes, n_classes):
        super(ResNet50, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.class_classifier = nn.Linear(512 * Bottleneck.expansion, n_classes)


class AlexNet(BasicNet):
    def __init__(self, domain_classes, n_classes, gen):
        super(AlexNet, self).__init__()
        pretrained = alexnet(pretrained=True)
        self.build_self(pretrained, n_classes)

    def build_self(self, pretrained, n_classes):
        self._convs = pretrained.features
        self._classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(),
            pretrained.classifier[1],  # nn.Linear(256 * 6 * 6, 4096),  #
            nn.ReLU(inplace=True),
            nn.Dropout(),
            pretrained.classifier[4],  # nn.Linear(4096, 4096),  #
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(self._convs, self._classifier)
        self.class_classifier = nn.Sequential(nn.Dropout(), nn.Linear(4096, n_classes))

    def get_trainable_params(self):
        return itertools.chain(self._classifier.parameters(), self.class_classifier.parameters())


class AlexNetADial(BasicNet):
    def __init__(self, domain_classes, n_classes, gen):
        super(AlexNetADial, self).__init__()
        pretrained = alexnet(pretrained=True)
        self.convs = pretrained.features
        self.fc6 = pretrained.classifier[1]  # nn.Linear(9216, 4096)
        self.dial6 = MSAutoDIAL(4096, domain_classes)
        self.fc7 = pretrained.classifier[4]  # nn.Linear(4096, 4096)
        self.dial7 = MSAutoDIAL(4096, domain_classes)
        self.class_classifier = nn.Sequential(#nn.Dropout(),
                                              nn.Linear(4096, n_classes))

    def forward(self, input_data, domain):
        x = self.convs(input_data)
        x = x.view(x.size(0), -1)
        #x = self.fc6(F.dropout(x))
        x = self.fc6(x)
        x = F.relu(self.dial6(x, domain))
        #x = self.fc7(F.dropout(x))
        x = self.fc7(x)
        x = F.relu(self.dial7(x, domain))
        return self.class_classifier(x)

    def get_trainable_params(self):
        return itertools.chain(self.fc6.parameters(), self.fc7.parameters(), self.dial6.parameters(), self.dial7.parameters(),
                               self.class_classifier.parameters())


# class test_AlexCaffeDial(nn.Module):
#     def __init__(self, domain_classes, n_classes, gen):
#         super(AlexCaffeDial, self).__init__()
#         dropout=False
#         self.features = nn.Sequential(OrderedDict([
#             ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
#             ("relu1", nn.ReLU(inplace=True)),
#             ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
#             ("relu2", nn.ReLU(inplace=True)),
#             ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
#             ("relu3", nn.ReLU(inplace=True)),
#             ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
#             ("relu4", nn.ReLU(inplace=True)),
#             ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
#             ("relu5", nn.ReLU(inplace=True)),
#             ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#         ]))
#         self.classifier = nn.Sequential(OrderedDict([
#             ("fc6", nn.Linear(256 * 6 * 6, 4096)),
#             ("relu6", nn.ReLU(inplace=True)),
#             ("drop6", nn.Dropout() if dropout else Id()),
#             ("dial6", MSAutoDIAL(4096, domain_classes)),
#             ("fc7", nn.Linear(4096, 4096)),
#             ("relu7", nn.ReLU(inplace=True)),
#             ("dial7", MSAutoDIAL(4096, domain_classes)),
#             ("drop7", nn.Dropout() if dropout else Id()),
#             ("fc8", nn.Linear(4096, n_classes))
#         ])
#         state_dict = torch.load("models/alexnet_caffe.pth.tar")          
#         # del state_dict["classifier.fc7.weight"]
#         # del state_dict["classifier.fc7.bias"]
#         # del state_dict["classifier.fc8.weight"]
#         # del state_dict["classifier.fc8.bias"]
#         self.load_state_dict(state_dict)
#         nn.init.xavier_uniform_(self.classifier[-1].weight, .1)
#         nn.init.constant_(self.classifier[-1].bias, 0.)

#     def get_trainable_params(self):
#         return self.classifier.parameters()
    
#     def forward(self, input_data, domain):
#         x = self.features(input_data)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
    
class MyAlexCaffe(AlexNetCaffe):
    def get_trainable_params(self):
        return self.classifier.parameters()
    def forward(self, input_data, domain):
        x = self.features(input_data)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
                                        
class AlexNetCaffeADial(BasicNet):
    def __init__(self, domain_classes, n_classes, gen):
        super(AlexNetCaffeADial, self).__init__()
        model = MyAlexCaffe(dropout=False)
        model.load_state_dict(torch.load("models/alexnet_caffe.pth.tar"))
        model.classifier[-1] = nn.Linear(4096, n_classes)
        nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
        nn.init.constant_(model.classifier[-1].bias, 0.)
        pretrained = model
        self.convs = pretrained.features
        self.fc6 = pretrained.classifier[0]  # nn.Linear(9216, 4096)
        self.dial6 = MSAutoDIAL(4096, domain_classes)
        self.fc7 = pretrained.classifier[3]  # nn.Linear(4096, 4096)
        self.dial7 = MSAutoDIAL(4096, domain_classes)
        self.class_classifier = pretrained.classifier[6]

    def forward(self, input_data, domain):
        x = self.convs(input_data)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = F.relu(self.dial6(x, domain))
        x = self.fc7(x)
        x = F.relu(self.dial7(x, domain))
        return self.class_classifier(x)

    def get_trainable_params(self):
        return itertools.chain(self.fc7.parameters(), self.dial6.parameters(), self.dial7.parameters(), #self.fc6.parameters(), 
                               self.class_classifier.parameters())

                                        
def get_alex_caffe(domain_classes, n_classes, gen):
    model = MyAlexCaffe(dropout=False)
    model.load_state_dict(torch.load("models/alexnet_caffe.pth.tar"))
    model.classifier[-1] = nn.Linear(4096, n_classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
    nn.init.constant_(model.classifier[-1].bias, 0.)
    return model


class CaffenetADial(AlexNetADial):
    def __init__(self, domain_classes, n_classes, gen):
        super(CaffenetADial, self).__init__(domain_classes, n_classes, gen)
        pretrained = load_caffenet()
        self.convs = pretrained[:15]
        self.fc6 = pretrained[16]
        self.dial6 = MSAutoDIAL(4096, domain_classes)
        self.fc7 = pretrained[19]
        self.dial7 = MSAutoDIAL(4096, domain_classes)
        self.class_classifier = nn.Linear(4096, n_classes)

    def forward(self, input_data, domain):
        x = input_data[:,(2,1,0)]
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        # x = F.dropout(x)
        x = self.fc6(x)
        x = F.relu(self.dial6(x, domain))
        # x = F.dropout(x)
        x = self.fc7(x)
        x = F.relu(self.dial7(x, domain))
        x = F.dropout(x)
        return self.class_classifier(x)


class AlexNetNoBottleneck(BasicNet):
    def __init__(self, domain_classes, n_classes):
        super(AlexNetNoBottleneck, self).__init__()
        pretrained = alexnet(pretrained=True)
        self._convs = pretrained.features
        self._classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(),
            pretrained.classifier[1],  # nn.Linear(256 * 6 * 6, 4096),  #
            nn.ReLU(inplace=True),
            nn.Dropout(),
            pretrained.classifier[4],  # nn.Linear(4096, 4096),  #
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(self._convs, self._classifier)
        self.class_classifier = nn.Linear(4096, n_classes)
        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 1024),  # pretrained.classifier[1]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),  # pretrained.classifier[4]
            nn.ReLU(inplace=True),
            nn.Linear(1024, domain_classes),
        )

    def get_trainable_params(self):
        return itertools.chain(self.domain_classifier.parameters(), self.class_classifier.parameters())
