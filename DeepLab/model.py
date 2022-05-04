import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class SqueezeAndExcite(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SqueezeAndExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv_3x3_d_6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv_3x3_d_12 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv_3x3_d_18 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.adp_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(512, 256, kernel_size=1, bias=False)

    def forward(self, x):
        x_h = x.size()[2]
        x_w = x.size()[3]
        conv_1x1_1 = self.conv_1x1_1(x)
        conv_3x3_d_6 = self.conv_3x3_d_6(x)
        conv_3x3_d_12 = self.conv_3x3_d_12(x)
        conv_3x3_d_18 = self.conv_3x3_d_18(x)
        image_pooling = F.relu(self.last_conv(self.adp_pool(x)))
        image_pooling = F.interpolate(image_pooling, size=(x_h, x_w), mode="bilinear")
        return conv_1x1_1, conv_3x3_d_6, conv_3x3_d_12, conv_3x3_d_18, image_pooling


class DeepLab(nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()

        self.resnet = models.resnet34(pretrained=True)
        self.aspp = ASPP()

        self.low_conv = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        self.aspp_conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeAndExcite(256))

        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1)
        )

        self.squeezeAndExcite = SqueezeAndExcite(304)

    def forward(self, x):
        x_h = x.size()[2]
        x_w = x.size()[3]

        self.resnet.layer4.register_forward_hook(get_activation('layer4'))
        self.resnet.layer1.register_forward_hook(get_activation('layer1'))
        x = self.resnet(x)

        low_features = activation['layer1']
        low_features = self.low_conv(low_features)

        low_h = low_features.size()[2]
        low_w = low_features.size()[3]

        x = activation['layer4']
        conv_1x1_1, conv_3x3_d_6, conv_3x3_d_12, conv_3x3_d_18, image_pooling = self.aspp(x)
        x = torch.cat([conv_1x1_1, conv_3x3_d_6, conv_3x3_d_12, conv_3x3_d_18, image_pooling], 1)
        x = self.aspp_conv(x)

        x = F.interpolate(x, size=(low_h, low_w), mode="bilinear")

        x = torch.cat([x, low_features], 1)
        del low_features

        x = self.squeezeAndExcite(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(x_h, x_w), mode="bilinear")
        x = self.last_conv(x)
        return x


def test():
    x = torch.randn((1, 3, 512, 512))
    model = DeepLab()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
