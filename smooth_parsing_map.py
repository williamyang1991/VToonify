import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import cv2
import math
import argparse
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from model.raft.core.raft import RAFT
from model.raft.core.utils.utils import InputPadder
from model.bisenet.model import BiSeNet
from model.stylegan.model import Downsample

class Options():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Smooth Parsing Maps")
        self.parser.add_argument("--window_size", type=int, default=5, help="temporal window size")
        
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--raft_path", type=str, default='./checkpoint/raft-things.pth', help="path of the RAFT model")
        
        self.parser.add_argument("--video_path", type=str, help="path of the target video")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output parsing maps")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt                

# from RAFT
def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()


    #x = x.cuda()
    grid = grid.cuda()
    vgrid = grid + flo # B,2,H,W

    # scale grid to [-1,1] 
    ##2019 code
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

    ##2019 author
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

     ##2019 code
     # mask = torch.floor(torch.clamp(mask, 0 ,1))

    return output*mask, mask

    
if __name__ == "__main__":

    parser = Options()
    args = parser.parse()
    print('*'*98)
    
    
    device = "cuda"
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    raft_model = torch.nn.DataParallel(RAFT(parser.parse_args(['--model', args.raft_path])))
    raft_model.load_state_dict(torch.load(args.raft_path))

    raft_model = raft_model.module
    raft_model.to(device)
    raft_model.eval()

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    down = Downsample(kernel=[1, 3, 3, 1], factor=2).to(device).eval()

    print('Load models successfully!')
    
    window = args.window_size

    video_cap = cv2.VideoCapture(args.video_path)
    num = int(video_cap.get(7))

    Is = []
    for i in range(num):
        success, frame = video_cap.read()
        if success == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            Is += [transform(frame).unsqueeze(dim=0).cpu()]
    video_cap.release()      

    # enlarge frames for more accurate parsing maps and optical flows     
    Is = F.upsample(torch.cat(Is, dim=0), scale_factor=2, mode='bilinear')
    Is_ = torch.cat((Is[0:window], Is, Is[-window:]), dim=0)

    print('Load video with %d frames successfully!'%(len(Is)))

    Ps = []
    for i in tqdm(range(len(Is))):
        with torch.no_grad():
            Ps += [parsingpredictor(2*Is[i:i+1].to(device))[0].detach().cpu()]
    Ps = torch.cat(Ps, dim=0)
    Ps_ = torch.cat((Ps[0:window], Ps, Ps[-window:]), dim=0)

    print('Predict parsing maps successfully!')
    
    
    # temporal weights of the (2*args.window_size+1) frames
    wt = torch.exp(-(torch.arange(2*window+1).float()-window)**2/(2*((window+0.5)**2))).reshape(2*window+1,1,1,1).to(device)
    
    parse = []
    for ii in tqdm(range(len(Is))):
        i = ii + window
        image2 = Is_[i-window:i+window+1].to(device)
        image1 = Is_[i].repeat(2*window+1,1,1,1).to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            flow_low, flow_up = raft_model((image1+1)*255.0/2, (image2+1)*255.0/2, iters=20, test_mode=True)
            output, mask = warp(torch.cat((image2, Ps_[i-window:i+window+1].to(device)), dim=1), flow_up)
            aligned_Is = output[:,0:3].detach()
            aligned_Ps = output[:,3:].detach()
            # the spatial weight
            ws = torch.exp(-((aligned_Is-image1)**2).mean(dim=1, keepdims=True)/(2*(0.2**2))) * mask[:,0:1]
            aligned_Ps[window] = Ps_[i].to(device)
            # the weight between i and i shoud be 1.0
            ws[window,:,:,:] = 1.0
            weights = ws*wt
            weights = weights / weights.sum(dim=(0), keepdims=True)
            fused_Ps = (aligned_Ps * weights).sum(dim=0, keepdims=True)
            parse += [down(fused_Ps).detach().cpu()]
    parse = torch.cat(parse, dim=0)
    
    basename = os.path.basename(args.video_path).split('.')[0]
    np.save(os.path.join(args.output_path, basename+'_parsingmap.npy'), parse.numpy())
    
    print('Done!')