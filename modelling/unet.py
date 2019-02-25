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
        #layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
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
    """ U-Net architecture definition
    """
    def __init__(self, num_classes, num_channels, for_fcn, use_planet, resize_planet):
        super(UNet, self).__init__()

        self.use_planet = use_planet
        self.resize_planet = resize_planet

        self.for_fcn = for_fcn
        self.downsample = _DownSample() 
        
        feats = 32
        self.enc1 = _EncoderBlock(num_channels, feats)
        self.enc2 = _EncoderBlock(feats, feats*2)
        
        # WITH PLANET 256 x 256
        #self.enc3 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(num_channels, feats*4)
        self.enc4 = _EncoderBlock(feats*4, feats*8)

        self.center = nn.Sequential(
            nn.Conv2d(feats*8, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True))
        self.center_decode = nn.Sequential(
            nn.Conv2d(feats*16, feats*16, kernel_size=3, padding=1),
            nn.GroupNorm(feats*16 // 16, feats*16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(feats*16, feats*8, kernel_size=2, stride=2))    
        self.dec4 = _DecoderBlock(feats*16, feats*8, feats*4)
        self.dec3 = nn.Sequential(
            nn.Conv2d(feats*8, feats*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feats*4, feats*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(inplace=True)
        )

        self.final = nn.Conv2d(feats*2, num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()
        initialize_weights(self)

        # WITH PLANET RESIZED
        #self.center = nn.Sequential(
        #    nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #    nn.GroupNorm(128 // 16, 128),
        #    nn.LeakyReLU(inplace=True))
        #self.center_decode = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #    nn.GroupNorm(128 // 16, 128),
        #    nn.LeakyReLU(inplace=True),
        #    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        #self.dec2 = _DecoderBlock(128, 64, 32)
        #self.dec1 = nn.Sequential(
        #    nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(32),
        #    nn.LeakyReLU(inplace=True),
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(32),
        #    nn.LeakyReLU(inplace=True),
        #)
        #self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        #self.softmax = nn.Softmax2d()
        #initialize_weights(self)

    def forward(self, x):
        x = x.cuda()
        print('x; ', x.shape)
        
        # WITH PLANET 256 x 256
        #enc1 = self.enc1(x)
        #down1 = self.downsample(enc1)
        #print('enc1; ', enc1.shape)
        #enc2 = self.enc2(down1)
        #down2 = self.downsample(enc2)
        #print('enc2; ', enc2.shape)
        #enc3 = self.enc3(down2)
        enc3 = self.enc3(x)
        down3 = self.downsample(enc3)
        print('enc3; ', enc3.shape)
        enc4 = self.enc4(down3)
        down4 = self.downsample(enc4)
        print('enc4; ', enc4.shape)
        center1 = self.center(down4)
        print('center1; ', center1.shape)
        center2 = self.center_decode(center1)
        print('center2; ', center2.shape)
        dec4 = self.dec4(torch.cat([center2, enc4], 1)) 
        print('dec4; ', dec4.shape)
        dec3 = self.dec3(torch.cat([dec4, enc3], 1)) 
        print('dec3; ', dec3.shape)
        final = self.final(dec3)
        print('final: ', final.shape)


        # WITH PLANET RESIZED
        #enc1 = self.enc1(x)
        #down1 = self.downsample(enc1)
        #print('enc1; ', enc1.shape)
        #enc2 = self.enc2(down1)
        #down2 = self.downsample(enc2)
        #print('enc2; ', enc2.shape)
        #center1 = self.center(down2)
        #print('center1; ', center1.shape)
        #center2 = self.center_decode(center1)
        #print('center2; ', center2.shape)
        #dec2 = self.dec2(torch.cat([center2, enc2], 1)) 
        #print('dec2; ', dec2.shape)
        #dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        #print('dec1; ', dec1.shape)
        #final = self.final(dec1)
        #print('final: ', final.shape)

        if self.for_fcn:
            return final
        else:
            final = self.softmax(final)
            final = torch.log(final)
            return final
