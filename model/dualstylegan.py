import random
import torch
from torch import nn
from model.stylegan.model import ConvLayer, PixelNorm, EqualLinear, Generator

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, fin, style_dim=512):
        super().__init__()

        self.norm = nn.InstanceNorm2d(fin, affine=False)
        self.style = nn.Linear(style_dim, fin * 2)

        self.style.bias.data[:fin] = 1
        self.style.bias.data[fin:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

# modulative residual blocks (ModRes)
class AdaResBlock(nn.Module):
    def __init__(self, fin, style_dim=512, dilation=1): # modified
        super().__init__()

        self.conv = ConvLayer(fin, fin, 3, dilation=dilation) # modified
        self.conv2 = ConvLayer(fin, fin, 3, dilation=dilation) # modified
        self.norm = AdaptiveInstanceNorm(fin, style_dim)
        self.norm2 = AdaptiveInstanceNorm(fin, style_dim)
        
        # model initialization
        # the convolution filters are set to values close to 0 to produce negligible residual features
        self.conv[0].weight.data *= 0.01
        self.conv2[0].weight.data *= 0.01
        
    def forward(self, x, s, w=1):
        skip = x
        if w == 0:
            return skip
        out = self.conv(self.norm(x, s))
        out = self.conv2(self.norm2(out, s))
        out = out * w + skip
        return out

class DualStyleGAN(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, twoRes=True, res_index=6):
        super().__init__()
        
        layers = [PixelNorm()]
        for i in range(n_mlp-6):
            layers.append(EqualLinear(512, 512, lr_mul=0.01, activation="fused_lrelu"))
        # color transform blocks T_c
        self.style = nn.Sequential(*layers)
        # StyleGAN2
        self.generator = Generator(size, style_dim, n_mlp, channel_multiplier) 
        # The extrinsic style path
        self.res = nn.ModuleList()
        self.res_index = res_index//2 * 2
        self.res.append(AdaResBlock(self.generator.channels[2 ** 2])) # for conv1
        for i in range(3, self.generator.log_size + 1):
            out_channel = self.generator.channels[2 ** i]
            if i < 3 + self.res_index//2:
                # ModRes
                self.res.append(AdaResBlock(out_channel))
                self.res.append(AdaResBlock(out_channel))
            else:
                # structure transform block T_s
                self.res.append(EqualLinear(512, 512))
                # FC layer is initialized with identity matrices, meaning no changes to the input latent code
                self.res[-1].weight.data = torch.eye(512) * 512.0**0.5 + torch.randn(512, 512) * 0.01
                self.res.append(EqualLinear(512, 512)) 
                self.res[-1].weight.data = torch.eye(512) * 512.0**0.5 + torch.randn(512, 512) * 0.01   
        self.res.append(EqualLinear(512, 512)) # for to_rgb7
        self.res[-1].weight.data = torch.eye(512) * 512.0**0.5 + torch.randn(512, 512) * 0.01             
        self.size = self.generator.size
        self.style_dim = self.generator.style_dim
        self.log_size = self.generator.log_size
        self.num_layers = self.generator.num_layers
        self.n_latent = self.generator.n_latent
        self.channels = self.generator.channels
        
    def forward(
        self,
        styles, # intrinsic style code
        exstyles, # extrinsic style code
        return_latents=False, 
        return_feat=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        z_plus_latent=False, # intrinsic style code is z+ or z
        use_res=True,        # whether to use the extrinsic style path
        fuse_index=18,       # layers > fuse_index do not use the extrinsic style path
        interp_weights=[1]*18, # weight vector for style combination of two paths
    ):

        if not input_is_latent:
            if not z_plus_latent:
                styles = [self.generator.style(s) for s in styles]
            else:
                styles = [self.generator.style(s.reshape(s.shape[0]*s.shape[1], s.shape[2])).reshape(s.shape) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.generator.num_layers
            else:
                noise = [
                    getattr(self.generator.noises, f"noise_{i}") for i in range(self.generator.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t
        
        if len(styles) < 2:
            inject_index = self.generator.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.generator.n_latent - 1)

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.generator.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)
            else:
                latent = torch.cat([styles[0][:,0:inject_index], styles[1][:,inject_index:]], 1)
            
        if use_res:
            if exstyles.ndim < 3:
                resstyles = self.style(exstyles).unsqueeze(1).repeat(1, self.generator.n_latent, 1)
                adastyles = exstyles.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
            else:
                nB, nL, nD = exstyles.shape
                resstyles = self.style(exstyles.reshape(nB*nL, nD)).reshape(nB, nL, nD)
                adastyles = exstyles
        
        out = self.generator.input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=noise[0])
        if use_res and fuse_index > 0:
            out = self.res[0](out, resstyles[:, 0], interp_weights[0])
        
        skip = self.generator.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.generator.convs[::2], self.generator.convs[1::2], noise[1::2], noise[2::2], self.generator.to_rgbs):
            if use_res and fuse_index >= i and i > self.res_index:
                out = conv1(out, interp_weights[i] * self.res[i](adastyles[:, i]) + 
                            (1-interp_weights[i]) * latent[:, i], noise=noise1)
            else:
                out = conv1(out, latent[:, i], noise=noise1)
            if use_res and fuse_index >= i and i <= self.res_index:
                out = self.res[i](out, resstyles[:, i], interp_weights[i])
            if use_res and fuse_index >= (i+1) and i > self.res_index:
                out = conv2(out, interp_weights[i+1] * self.res[i+1](adastyles[:, i+1]) + 
                            (1-interp_weights[i+1]) * latent[:, i+1], noise=noise2)
            else:
                out = conv2(out, latent[:, i + 1], noise=noise2)
            if use_res and fuse_index >= (i+1) and i <= self.res_index:
                out = self.res[i+1](out, resstyles[:, i+1], interp_weights[i+1])   
            if use_res and fuse_index >= (i+2) and i >= self.res_index-1:
                skip = to_rgb(out, interp_weights[i+2] * self.res[i+2](adastyles[:, i+2]) +
                              (1-interp_weights[i+2]) * latent[:, i + 2], skip)
            else:
                skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
            if i > self.res_index and return_feat:
                return out, skip

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None  
        
    def make_noise(self):
        return self.generator.make_noise()

    def mean_latent(self, n_latent):
        return self.generator.mean_latent(n_latent)

    def get_latent(self, input):
        return self.generator.style(input)