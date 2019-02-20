import torch 
import torch.nn as nn
import torch.nn.functional as F

from modelling.util import initialize_weights


class _EncoderBlock(nn.Module):
    """ U-Net encoder block
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """ U-Net decoder block
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
    
class UNet(nn.Module):
    """ U-Net architecture definition
    """
    def __init__(self, num_classes, num_channels, for_fcn):
        super(UNet, self).__init__()
        self.for_fcn = for_fcn
        self.enc1 = _EncoderBlock(num_channels, 64)
        self.enc2 = _EncoderBlock(64, 128)
        # self.enc3 = _EncoderBlock(128, 256, dropout=True)
        # self.enc4 = _EncoderBlock(256, 512, dropout=True)
        # self.center = _DecoderBlock(512, 1024, 512)
        self.center = _DecoderBlock(128, 256, 128)
        # self.dec4 = _DecoderBlock(1024, 512, 256)
        # self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x):
        x = x.cuda()
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        # enc3 = self.enc3(enc2)
        # enc4 = self.enc4(enc3)
        # center = self.center(enc4)
        center = self.center(enc2)
        # dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([center, F.upsample(enc2, center.size()[2:], mode='bilinear')], 1))
        # dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        # dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        final = F.upsample(final, x.size()[2:], mode='bilinear')
        if self.for_fcn:
            return final
        else:
            final = self.softmax(final)
            final = torch.log(final)
            return final
