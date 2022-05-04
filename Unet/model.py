import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, mid_channels=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        temp_channel = in_channels
        for channel in mid_channels:
            self.downs.append(DoubleConv(temp_channel, channel))
            temp_channel = channel

        self.middle_conv = DoubleConv(mid_channels[-1], mid_channels[-1] * 2)

        for channel in reversed(mid_channels):
            self.ups.append(
                nn.ConvTranspose2d(
                    channel * 2, channel, kernel_size=2, stride=2, ))
            self.ups.append(DoubleConv(channel * 2, channel))

        self.out_conv = nn.Conv2d(mid_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        copied_data = []

        for down in self.downs:
            x = down(x)
            copied_data.append(x)
            x = self.pool(x)

        x = self.middle_conv(x)
        copied_data = copied_data[::-1]

        for i in range(len(self.ups)):
            if i % 2 == 0:
                x = self.ups[i](x)
            else:
                concat_data = copied_data[i // 2]
                if x.shape != concat_data.shape:
                    x = torchvision.transforms.functional.resize(x, size=concat_data.shape[2:])
                    # x = torchvision.transforms.functional.center_crop(x,concat_data.shape[2])
                concat_skip = torch.cat((concat_data, x), dim=1)
                x = self.ups[i](concat_skip)
        x = self.out_conv(x)
        return x


def test():
    x = torch.randn((1, 3, 256, 256))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
