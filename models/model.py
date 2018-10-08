import torch
import torch.nn.functional as func
from torch import nn as nn

from models.torch_future import Flatten


def get_classifier(name, domain_classes, n_classes, generalization):
    return classifier_list[name](domain_classes, n_classes, generalization)


def get_net(args):
    domain_classes = args.domain_classes
    my_net = get_classifier(args.classifier, domain_classes=domain_classes, n_classes=args.n_classes, generalization=args.generalization)

    for p in my_net.parameters():
        p.requires_grad = True
    print(my_net)
    return my_net


def entropy_loss(x):
    return torch.sum(-func.softmax(x, 1) * func.log_softmax(x, 1), 1).mean()


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.features = None
        self.class_classifier = None

    def forward(self, input_data, domain=None):
        feature = self.features(input_data)
        feature = feature.view(input_data.shape[0], -1)
        class_output = self.class_classifier(feature)
        return class_output

    def get_trainable_params(self):
        return self.parameters()


class MnistModel(BasicNet):
    def __init__(self, domain_classes, n_classes):
        super(MnistModel, self).__init__()
        print("Using LeNet")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, domain_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, n_classes)
        )


class SVHNModel(BasicNet):
    def __init__(self, domain_classes, n_classes):
        super(SVHNModel, self).__init__()
        print("Using SVHN")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.Dropout(0.1, True),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.Dropout(0.25, True),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.Dropout(0.25, True),
            nn.ReLU(True)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(1024, domain_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 3072),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(3072, 2048),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(2048, n_classes)
        )


class MultisourceModel(nn.Module):
    def __init__(self, domain_classes, n_classes, generalization):
        super(MultisourceModel, self).__init__()
        self.domains = domain_classes
        if generalization:
            self.domains -= 1
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.class_classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(1024, n_classes)
        )

    def forward(self, input_data, domain=None):
        feature = self.features(input_data)
        class_output = self.class_classifier(feature)
        return class_output


class MultisourceDIALModel(nn.Module):
    def __init__(self, domain_classes, n_classes, generalization):
        super(MultisourceDIALModel, self).__init__()
        self.domains = domain_classes
        if generalization:
            self.domains -= 1
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.class_classifier_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048)
        )

        self.class_classifier_2 = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(2048, 1024)
        )

        self.class_classifier_3 = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(1024, n_classes)
        )
        self.batch_norms_1 = nn.ModuleList([nn.BatchNorm1d(2048) for i in range(domain_classes)])
        self.batch_norms_2 = nn.ModuleList([nn.BatchNorm1d(1024) for i in range(domain_classes)])

    def forward(self, input_data, domain):
        x = self.features(input_data)
        x = self.class_classifier_1(x)
        x = self.batch_norms_1[domain](x)
        x = self.class_classifier_2(x)
        x = self.batch_norms_2[domain](x)
        class_output = self.class_classifier_3(x)
        return class_output


class MultisourceBNModel(nn.Module):
    def __init__(self, domain_classes, n_classes, generalization):
        super(MultisourceBNModel, self).__init__()
        self.domains = domain_classes
        if generalization:
            self.domains -= 1
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.class_classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(1024, n_classes)
        )

    def forward(self, input_data, domain=None):
        feature = self.features(input_data)
        class_output = self.class_classifier(feature)
        return class_output


from models.large_models import ResNet50, AlexNet, CaffenetADial, AlexNetNoBottleneck, AlexNetADial, get_alex_caffe, AlexNetCaffeADial

classifier_list = {"mnist": MnistModel,
                   "svhn": SVHNModel,
                   "multi": MultisourceModel,
                   "multi_bn": MultisourceBNModel,
                   "multi_dial": MultisourceDIALModel,
                   "alexnet": AlexNet,
                   "alex_caffe": get_alex_caffe,
                   "alex_caffe_dial": AlexNetCaffeADial,
                   "alexnet_dial": AlexNetADial,
                   "alexnet_no_bottleneck": AlexNetNoBottleneck,
                   "caffenet_dial": CaffenetADial,
                   "resnet50": ResNet50}
