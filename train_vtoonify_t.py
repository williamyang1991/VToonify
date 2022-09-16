import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import math
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from util import *
from model.stylegan import lpips
from model.stylegan.model import Generator, Downsample
from model.vtoonify import VToonify, ConditionalDiscriminator
from model.bisenet.model import BiSeNet
from model.simple_augment import random_apply_affine
from model.stylegan.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

# In the paper, --weight for each style is set as follows,
# cartoon: default
# caricature: default
# pixar: 1 1 1 1 1 1 1 1 1 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
# comic: 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1 1 1 1 1 1
# arcane: 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1 1 1 1 1 1

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train VToonify-T")
        self.parser.add_argument("--iter", type=int, default=2000, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=8, help="batch sizes for each gpus")
        self.parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--start_iter", type=int, default=0, help="start iteration")
        self.parser.add_argument("--save_every", type=int, default=30000, help="interval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=30000, help="when to start saving a checkpoint")
        self.parser.add_argument("--log_every", type=int, default=200, help="interval of saving an intermediate image result")
        
        self.parser.add_argument("--adv_loss", type=float, default=0.01, help="the weight of adv loss")
        self.parser.add_argument("--grec_loss", type=float, default=0.1, help="the weight of mse recontruction loss")
        self.parser.add_argument("--perc_loss", type=float, default=0.01, help="the weight of perceptual loss")
        self.parser.add_argument("--tmp_loss", type=float, default=1.0, help="the weight of temporal consistency loss")
        
        self.parser.add_argument("--encoder_path", type=str, default=None, help="path to the pretrained encoder model")    
        self.parser.add_argument("--direction_path", type=str, default='./checkpoint/directions.npy', help="path to the editing direction latents")
        self.parser.add_argument("--stylegan_path", type=str, default='./checkpoint/stylegan2-ffhq-config-f.pt', help="path to the stylegan model")
        self.parser.add_argument("--finetunegan_path", type=str, default='./checkpoint/cartoon/finetune-000600.pt', help="path to the finetuned stylegan model")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[1]*9+[0]*9, help="the weight for blending two models")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        
        self.parser.add_argument("--name", type=str, default='vtoonify_t_cartoon', help="saved model name")
        self.parser.add_argument("--pretrain", action="store_true", help="if true, only pretrain the encoder")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.encoder_path is None:
            self.opt.encoder_path = os.path.join('./checkpoint/', self.opt.name, 'pretrain.pt')
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt
    

# pretrain E of vtoonify. 
# We train E so that its the last-layer feature matches the original 8-th-layer input feature of G1
# See Model initialization in Sec. 4.1.2 for the detail
def pretrain(args, generator, g_optim, g_ema, parsingpredictor, down, directions, basemodel, device):
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    recon_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
    else:
        g_module = generator

    accum = 0.5 ** (32 / (10 * 1000))
    
    requires_grad(g_module.encoder, True)
    
    for idx in pbar:
        i = idx + args.start_iter
        
        if i > args.iter:
            print("Done!")
            break
            
        with torch.no_grad():
            # during pretraining, no geometric transformations are applied.
            noise_sample = torch.randn(args.batch, 512).cuda()
            ws_ = basemodel.style(noise_sample).unsqueeze(1).repeat(1,18,1) # random w
            ws_[:, 3:7] += directions[torch.randint(0, directions.shape[0], (args.batch,)), 3:7] # w''=w'=w+n
            img_gen, _ = basemodel([ws_], input_is_latent=True, truncation=0.5, truncation_latent=0) # image part of x'
            img_gen = torch.clamp(img_gen, -1, 1).detach() 
            img_gen512 = down(img_gen.detach())
            img_gen256 = down(img_gen512.detach()) # image part of x'_down
            mask512 = parsingpredictor(2*torch.clamp(img_gen512, -1, 1))[0] 
            real_input = torch.cat((img_gen256, down(mask512)/16.0), dim=1).detach() # x'_down
            # f_G1^(8)(w'')
            real_feat, real_skip = g_ema.generator([ws_], input_is_latent=True, return_feature_ind = 6, truncation=0.5, truncation_latent=0)
            real_feat = real_feat.detach()
            real_skip = real_skip.detach()
            
        # f_E^(last)(x'_down)
        fake_feat, fake_skip = generator(real_input, style=None, return_feat=True)

        # L_E in Eq.(1)
        recon_loss = F.mse_loss(fake_feat, real_feat) + F.mse_loss(fake_skip, real_skip)

        loss_dict["emse"] = recon_loss

        generator.zero_grad()
        recon_loss.backward()  
        g_optim.step()    
        
        accumulate(g_ema.encoder, g_module.encoder, accum)     

        loss_reduced = reduce_loss_dict(loss_dict)

        emse_loss_val = loss_reduced["emse"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"iter: {i:d}; emse: {emse_loss_val:.3f}"
                )
            )

            if ((i+1) >= args.save_begin and (i+1) % args.save_every == 0) or (i+1) == args.iter:
                if (i+1) == args.iter:
                    savename = f"checkpoint/%s/pretrain.pt"%(args.name)
                else:
                    savename = f"checkpoint/%s/pretrain-%05d.pt"%(args.name, i+1)
                torch.save(
                    {
                        #"g": g_module.encoder.state_dict(),
                        "g_ema": g_ema.encoder.state_dict(),
                    },
                    savename,
                )
                

# generate paired data and train vtoonify, see Sec. 4.1.2 for the detail
def train(args, generator, discriminator, g_optim, d_optim, g_ema, percept, parsingpredictor, down, pspencoder, directions, basemodel, device):
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, smoothing=0.01, ncols=120, dynamic_ncols=False)

    d_loss = torch.tensor(0.0, device=device)
    g_loss = torch.tensor(0.0, device=device)
    grec_loss = torch.tensor(0.0, device=device)
    gfeat_loss = torch.tensor(0.0, device=device)
    temporal_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    
    for idx in pbar:
        i = idx + args.start_iter
        
        if i > args.iter:
            print("Done!")
            break
        
        ###### This part is for data generation. Generate pair (x, y, w'') as in Fig. 5 of the paper
        with torch.no_grad():   
            noise_sample = torch.randn(args.batch, 512).cuda()
            wc = basemodel.style(noise_sample).unsqueeze(1).repeat(1,18,1) # random w
            wc[:, 3:7] += directions[torch.randint(0, directions.shape[0], (args.batch,)), 3:7] # w'=w+n
            wc = wc.detach()
            xc, _ = basemodel([wc], input_is_latent=True, truncation=0.5, truncation_latent=0) 
            xc = torch.clamp(xc, -1, 1).detach() # x'
            xl = pspencoder(F.adaptive_avg_pool2d(xc, 256))
            xl = basemodel.style(xl.reshape(xl.shape[0]*xl.shape[1], xl.shape[2])).reshape(xl.shape) # E_s(x'_down)
            xl = torch.cat((wc[:,0:7]*0.5, xl[:,7:18]), dim=1).detach() # w'' = concatenate w' and E_s(x'_down)
            xs, _ = g_ema.generator([xl], input_is_latent=True)
            xs = torch.clamp(xs, -1, 1).detach() # y'
            # during training, random geometric transformations are applied.
            imgs, _ = random_apply_affine(torch.cat((xc.detach(),xs), dim=1), 0.2, None)
            real_input1024 = imgs[:,0:3].detach() # image part of x
            real_input512 = down(real_input1024).detach()
            real_input256 = down(real_input512).detach()
            mask512 = parsingpredictor(2*real_input512)[0]
            mask256 = down(mask512).detach()
            mask = F.adaptive_avg_pool2d(mask512, 1024).detach() # parsing part of x
            real_output = imgs[:,3:].detach()  # y
            real_input = torch.cat((real_input256, mask256/16.0), dim=1) # x_down
            # for log, sample a fixed input-output pair (x_down, y, w'')
            if idx == 0 or i == 0:
                samplein = real_input.clone().detach()
                sampleout = real_output.clone().detach()
                samplexl = xl.clone().detach()
        
        ###### This part is for training discriminator
        
        requires_grad(g_module.encoder, False)
        requires_grad(g_module.fusion_out, False)
        requires_grad(g_module.fusion_skip, False)  
        requires_grad(discriminator, True)
        
        fake_output = generator(real_input, xl)
        fake_pred = discriminator(F.adaptive_avg_pool2d(fake_output, 256))
        real_pred = discriminator(F.adaptive_avg_pool2d(real_output, 256))
        
        # L_adv in Eq.(3)
        d_loss = d_logistic_loss(real_pred, fake_pred) * args.adv_loss 
        loss_dict["d"] = d_loss
        
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()    
        
        ###### This part is for training generator (encoder and fusion modules)
        
        requires_grad(g_module.encoder, True)
        requires_grad(g_module.fusion_out, True)
        requires_grad(g_module.fusion_skip, True)    
        requires_grad(discriminator, False)        

        fake_output = generator(real_input, xl)
        fake_pred = discriminator(F.adaptive_avg_pool2d(fake_output, 256))
        # L_adv in Eq.(3)
        g_loss = g_nonsaturating_loss(fake_pred) * args.adv_loss
        # L_rec in Eq.(2)
        grec_loss = F.mse_loss(fake_output, real_output) * args.grec_loss
        gfeat_loss = percept(F.adaptive_avg_pool2d(fake_output, 512), # 1024 will out of memory
                             F.adaptive_avg_pool2d(real_output, 512)).sum() * args.perc_loss # 256 will get blurry output
 
        loss_dict["g"] = g_loss
        loss_dict["gr"] = grec_loss
        loss_dict["gf"] = gfeat_loss

        w = random.randint(0,1024-896)
        h = random.randint(0,1024-896)
        crop_input = torch.cat((real_input1024[:,:,w:w+896,h:h+896], mask[:,:,w:w+896,h:h+896]/16.0), dim=1).detach()
        crop_input = down(down(crop_input))  
        crop_fake_output = fake_output[:,:,w:w+896,h:h+896]
        fake_crop_output = generator(crop_input, xl)
        # L_tmp in Eq.(4), gradually increase the weight of L_tmp
        temporal_loss = ((fake_crop_output-crop_fake_output)**2).mean() * max(idx/(args.iter/2.0)-1, 0) * args.tmp_loss
        loss_dict["tp"] = temporal_loss

        generator.zero_grad()
        (g_loss + grec_loss + gfeat_loss + temporal_loss).backward() 
        g_optim.step()        
        
        accumulate(g_ema.encoder, g_module.encoder, accum)
        accumulate(g_ema.fusion_out, g_module.fusion_out, accum)
        accumulate(g_ema.fusion_skip, g_module.fusion_skip, accum)        

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        gr_loss_val = loss_reduced["gr"].mean().item()
        gf_loss_val = loss_reduced["gf"].mean().item()
        tmp_loss_val = loss_reduced["tp"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"iter: {i:d}; advd: {d_loss_val:.3f}; advg: {g_loss_val:.3f}; mse: {gr_loss_val:.3f}; "
                    f"perc: {gf_loss_val:.3f}; tmp: {tmp_loss_val:.3f}"
                )
            )

            if i % args.log_every == 0 or (i+1) == args.iter:
                with torch.no_grad():
                    g_ema.eval()
                    sample = g_ema(samplein, samplexl)
                    sample = F.interpolate(torch.cat((sampleout, sample), dim=0), 256)
                    utils.save_image(
                        sample,
                        f"log/%s/%05d.jpg"%(args.name, i),
                        nrow=int(args.batch),
                        normalize=True,
                        range=(-1, 1),
                    )

            if ((i+1) >= args.save_begin and (i+1) % args.save_every == 0) or (i+1) == args.iter:
                if (i+1) == args.iter:
                    savename = f"checkpoint/%s/vtoonify.pt"%(args.name)
                else:
                    savename = f"checkpoint/%s/vtoonify_%05d.pt"%(args.name, i+1)                
                torch.save(
                    {
                        #"g": g_module.state_dict(),
                        #"d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                    },
                    savename,
                )
                
                

if __name__ == "__main__":
    
    device = "cuda"
    parser = TrainOptions()  
    args = parser.parse()
    if args.local_rank == 0:
        print('*'*98)
        if not os.path.exists("log/%s/"%(args.name)):
            os.makedirs("log/%s/"%(args.name))
        if not os.path.exists("checkpoint/%s/"%(args.name)):
            os.makedirs("checkpoint/%s/"%(args.name))
        
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    generator = VToonify(backbone = 'toonify').to(device)
    generator.apply(weights_init)
    g_ema = VToonify(backbone = 'toonify').to(device)
    g_ema.eval()

    basemodel = Generator(1024, 512, 8, 2).to(device) # G0
    finetunemodel = Generator(1024, 512, 8, 2).to(device) 
    basemodel.load_state_dict(torch.load(args.stylegan_path, map_location=lambda storage, loc: storage)['g_ema'])
    finetunemodel.load_state_dict(torch.load(args.finetunegan_path, map_location=lambda storage, loc: storage)['g_ema'])
    fused_state_dict = blend_models(finetunemodel, basemodel, args.weight) # G1
    generator.generator.load_state_dict(fused_state_dict) # load G1
    g_ema.generator.load_state_dict(fused_state_dict)
    requires_grad(basemodel, False)
    requires_grad(generator.generator, False)
    requires_grad(g_ema.generator, False)

    if not args.pretrain:
        generator.encoder.load_state_dict(torch.load(args.encoder_path, map_location=lambda storage, loc: storage)["g_ema"])
        # we initialize the fusion modules to map f_G \otimes f_E to f_G.
        for k in generator.fusion_out:
            k.weight.data *= 0.01
            k.weight[:,0:k.weight.shape[0],1,1].data += torch.eye(k.weight.shape[0]).cuda()
        for k in generator.fusion_skip:
            k.weight.data *= 0.01
            k.weight[:,0:k.weight.shape[0],1,1].data += torch.eye(k.weight.shape[0]).cuda()

    accumulate(g_ema.encoder, generator.encoder, 0)
    accumulate(g_ema.fusion_out, generator.fusion_out, 0)
    accumulate(g_ema.fusion_skip, generator.fusion_skip, 0) 

    g_parameters = list(generator.encoder.parameters()) 
    if not args.pretrain:
        g_parameters = g_parameters + list(generator.fusion_out.parameters()) + list(generator.fusion_skip.parameters())

    g_optim = optim.Adam(
        g_parameters,
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()
    requires_grad(parsingpredictor, False)

    # we apply gaussian blur to the images to avoid flickers caused during downsampling
    down = Downsample(kernel=[1, 3, 3, 1], factor=2).to(device)
    requires_grad(down, False)

    directions = torch.tensor(np.load(args.direction_path)).to(device) 

    if not args.pretrain:
        discriminator = ConditionalDiscriminator(256).to(device)

        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
        )    

        if args.distributed:
            discriminator = nn.parallel.DistributedDataParallel(
                discriminator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[args.local_rank])
        requires_grad(percept.model.net, False)

        pspencoder = load_psp_standalone(args.style_encoder_path, device)  
    
    if args.local_rank == 0:
        print('Load models and data successfully loaded!')

    if args.pretrain:
        pretrain(args, generator, g_optim, g_ema, parsingpredictor, down, directions, basemodel, device)
    else:
        train(args, generator, discriminator, g_optim, d_optim, g_ema, percept, parsingpredictor, down, pspencoder, directions, basemodel, device)
