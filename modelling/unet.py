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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DownSample(nn.Module):
    """ U-Net downsample block
    """
    def __init__(self):
        super(_DownSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.downsample(x)

class _DecoderBlock(nn.Module):
    """ U-Net decoder block
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
class UNet(nn.Module):
    """Bring the encoder and decoder together into full UNet model
    """
    def __init__(self, num_classes, num_channels, late_feats_for_fcn=False, use_planet=False, resize_planet=False):
        super(UNet, self).__init__()
        self.unet_encode = UNet_Encode(num_channels, use_planet, resize_planet)
        self.unet_decode = UNet_Decode(num_classes, late_feats_for_fcn) 

    def forward(self, x):
        center1, enc4, enc3 = self.unet_encode(x)
        final = self.unet_decode(center1, enc4, enc3)
        return final

class UNet_Encode(nn.Module):
    """ U-Net architecture definition for encoding (first half of the "U")
    """
    def __init__(self, num_channels, use_planet=False, resize_planet=False):
        super(UNet_Encode, self).__init__()

        self.downsample = _DownSample() 
        self.use_planet = use_planet
        self.resize_planet = resize_planet      
  
        feats = 16
        if self.use_planet and self.resize_planet:
            enc3_infeats = num_channels
        elif not self.use_planet:
            enc3_infeats = num_channels
        else:
            enc3_infeats = feats*2
            self.enc1 = _EncoderBlock(num_channels, feats)
            self.enc2 = _EncoderBlock(feats, feats*2)
        
        self.enc3 = _EncoderBlock(enc3_infeats, feats*4)
        self.enc4 = _EncoderBlock(feats*4, feats*8)

        self.center = nn.Sequential(
            nn.Conv2d(feats*8, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True))
        
        initialize_weights(self)

    def forward(self, x):

        # ENCODE
        x = x.cuda()

        if self.use_planet and self.resize_planet:
            enc3 = self.enc3(x)
        elif not self.use_planet:
            enc3 = self.enc3(x)
        else:
            enc1 = self.enc1(x)
            down1 = self.downsample(enc1)
            enc2 = self.enc2(down1)
            down2 = self.downsample(enc2)
            enc3 = self.enc3(down2)

        down3 = self.downsample(enc3)
        enc4 = self.enc4(down3)
        down4 = self.downsample(enc4)
        center1 = self.center(down4)
        
        return center1, enc4, enc3

class UNet_Decode(nn.Module):
    """ U-Net architecture definition for decoding (second half of the "U")
    """
    def __init__(self, num_classes, late_feats_for_fcn):
        super(UNet_Decode, self).__init__()

        self.downsample = _DownSample() 
        self.late_feats_for_fcn = late_feats_for_fcn

        feats = 16
        self.center_decode = nn.Sequential(
            nn.Conv2d(feats*16, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(feats*16, feats*8, kernel_size=2, stride=2))    
        self.dec4 = _DecoderBlock(feats*16, feats*8, feats*4)
        self.final = nn.Sequential(
            nn.Conv2d(feats*8, feats*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*4, feats*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*2, num_classes, kernel_size=3, padding=1),
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)
        initialize_weights(self)

    def forward(self, center1, enc4, enc3):

        # DECODE
        center2 = self.center_decode(center1)
        dec4 = self.dec4(torch.cat([center2, enc4], 1)) 
        final = self.final(torch.cat([dec4, enc3], 1)) 

        if self.late_feats_for_fcn:
            return final
        else:
            final = self.logsoftmax(final)
            return final
