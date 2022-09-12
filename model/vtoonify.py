import torch
import numpy as np
import math
from torch import nn
from model.stylegan.model import ConvLayer, EqualLinear, Generator, ResBlock
from model.dualstylegan import AdaptiveInstanceNorm, AdaResBlock, DualStyleGAN
import torch.nn.functional as F

# IC-GAN: stylegan discriminator    
class ConditionalDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], use_condition=False, style_num=None):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1
        self.use_condition = use_condition
        
        if self.use_condition:
            self.condition_dim = 128
            # map style degree to 64-dimensional vector
            self.label_mapper = nn.Sequential(
                nn.Linear(1, 64),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(64, 64),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(64, self.condition_dim//2),
            )
            # map style code index to 64-dimensional vector
            self.style_mapper = nn.Embedding(style_num, self.condition_dim-self.condition_dim//2)
        else:
            self.condition_dim = 1
            
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], self.condition_dim),
        )
        
    def forward(self, input, degree_label=None, style_ind=None):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        
        if self.use_condition:
            h = self.final_linear(out)
            condition = torch.cat((self.label_mapper(degree_label), self.style_mapper(style_ind)), dim=1)
            out = (h * condition).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.condition_dim))
        else:
            out = self.final_linear(out)
            
        return out 
    

class VToonifyResBlock(nn.Module):
    def __init__(self, fin):
        super().__init__()

        self.conv = nn.Conv2d(fin, fin, 3,  1, 1)
        self.conv2 = nn.Conv2d(fin, fin, 3,  1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = self.lrelu(self.conv2(out))      
        out = (out + x) / math.sqrt(2)
        return out    

class Fusion(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # create conv layers
        self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1, bias=True)
        self.norm = AdaptiveInstanceNorm(in_channels + skip_channels, 128)
        self.conv2 = nn.Conv2d(in_channels + skip_channels, 1, 3, 1, 1, bias=True)
        #'''
        self.linear = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, f_G, f_E, d_s=1):
        # label of style degree
        label = self.linear(torch.zeros(f_G.size(0),1).to(f_G.device) + d_s)
        out = torch.cat([f_G, abs(f_G-f_E)], dim=1)
        m_E = (F.relu(self.conv2(self.norm(out, label)))).tanh()
        f_out = self.conv(torch.cat([f_G, f_E * m_E], dim=1))
        return f_out, m_E
    
class VToonify(nn.Module):
    def __init__(self,
                 in_size=256,
                 out_size=1024,
                 img_channels=3,
                 style_channels=512,
                 num_mlps=8,
                 channel_multiplier=2,
                 num_res_layers=6,
                 backbone = 'dualstylegan',
                ):

        super().__init__()

        self.backbone = backbone
        if self.backbone == 'dualstylegan':
            # DualStyleGAN, with weights being fixed
            self.generator = DualStyleGAN(out_size, style_channels, num_mlps, channel_multiplier)
        else:
            # StyleGANv2, with weights being fixed
            self.generator = Generator(out_size, style_channels, num_mlps, channel_multiplier)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 4, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(img_channels+19, 32, 3, 1, 1, bias=True), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(32, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        
        for res in encoder_res:
            in_channels = channels[res]
            if res > 32:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
                self.encoder.append(block)
            else:
                layers = []
                for _ in range(num_res_layers):
                    layers.append(VToonifyResBlock(in_channels))
                self.encoder.append(nn.Sequential(*layers))
                block = nn.Conv2d(in_channels, img_channels, 1, 1, 0, bias=True)
                self.encoder.append(block)
        
        # trainable fusion module
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            if self.backbone == 'dualstylegan':
                self.fusion_out.append(
                    Fusion(num_channels, num_channels, num_channels))
            else:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))

            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))
        
        # Modified ModRes blocks in DualStyleGAN, with weights being fixed
        if self.backbone == 'dualstylegan':
            self.res = nn.ModuleList()
            self.res.append(AdaResBlock(self.generator.channels[2 ** 2])) # for conv1, no use in this model
            for i in range(3, 6):
                out_channel = self.generator.channels[2 ** i]
                self.res.append(AdaResBlock(out_channel, dilation=2**(5-i)))
                self.res.append(AdaResBlock(out_channel, dilation=2**(5-i)))

    
    def forward(self, x, style, d_s=None, return_mask=False, return_feat=False):
        # map style to W+ space
        if style is not None and style.ndim < 3:
            if self.backbone == 'dualstylegan':
                resstyles = self.generator.style(style).unsqueeze(1).repeat(1, self.generator.n_latent, 1)
            adastyles = style.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
        elif style is not None:
            nB, nL, nD = style.shape
            if self.backbone == 'dualstylegan':
                resstyles = self.generator.style(style.reshape(nB*nL, nD)).reshape(nB, nL, nD)
            adastyles = style
        if self.backbone == 'dualstylegan':
            adastyles = adastyles.clone()
            for i in range(7, self.generator.n_latent):
                adastyles[:, i] = self.generator.res[i](adastyles[:, i])

        # obtain multi-scale content features
        feat = x
        encoder_features = []
        # downsampling conv parts of E
        for block in self.encoder[:-2]:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]
        # Resblocks in E
        for ii, block in enumerate(self.encoder[-2]):
            feat = block(feat)
            # adjust Resblocks with ModRes blocks
            if self.backbone == 'dualstylegan':
                feat = self.res[ii+1](feat, resstyles[:, ii+1], d_s)
        # the last-layer feature of E (inputs of backbone)
        out = feat
        skip = self.encoder[-1](feat)
        if return_feat:
            return out, skip
        
        # 32x32 ---> higher res
        _index = 1
        m_Es = []
        for conv1, conv2, to_rgb in zip(
            self.stylegan().convs[6::2], self.stylegan().convs[7::2], self.stylegan().to_rgbs[3:]): 
            
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (5+((_index-1)//2)) <= self.in_size:
                fusion_index = (_index - 1) // 2
                f_E = encoder_features[fusion_index]

                if self.backbone == 'dualstylegan':
                    out, m_E = self.fusion_out[fusion_index](out, f_E, d_s)
                    skip = self.fusion_skip[fusion_index](torch.cat([skip, f_E*m_E], dim=1))
                    m_Es += [m_E]
                else:
                    out = self.fusion_out[fusion_index](torch.cat([out, f_E], dim=1))
                    skip = self.fusion_skip[fusion_index](torch.cat([skip, f_E], dim=1))  
            
            # remove the noise input
            batch, _, height, width = out.shape
            noise = x.new_empty(batch, 1, height * 2, width * 2).normal_().detach() * 0.0
            
            out = conv1(out, adastyles[:, _index+6], noise=noise)
            out = conv2(out, adastyles[:, _index+7], noise=noise)
            skip = to_rgb(out, adastyles[:, _index+8], skip)
            _index += 2

        image = skip
        if return_mask and self.backbone == 'dualstylegan':
            return image, m_Es
        return image
    
    def stylegan(self):
        if self.backbone == 'dualstylegan':
            return self.generator.generator
        else:
            return self.generator
        
    def zplus2wplus(self, zplus):
        return self.stylegan().style(zplus.reshape(zplus.shape[0]*zplus.shape[1], zplus.shape[2])).reshape(zplus.shape)