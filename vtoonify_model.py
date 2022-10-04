# this code is for web demo in colab notebook

from __future__ import annotations
import gradio as gr
import pathlib
import sys

from util import load_psp_standalone, get_video_crop_parameter, tensor2cv2
import torch
import torch.nn as nn
import numpy as np
import dlib
import cv2
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
import torch.nn.functional as F
from torchvision import transforms
from model.encoder.align_all_parallel import align_face
import gc
import huggingface_hub
import os

MODEL_REPO = 'PKUWilliamYang/VToonify'

class Model():
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.style_types = {
            'cartoon1': ['vtoonify_d_cartoon/vtoonify_s026_d0.5.pt', 26],
            'cartoon1-d': ['vtoonify_d_cartoon/vtoonify_s_d.pt', 26],
            'cartoon2-d': ['vtoonify_d_cartoon/vtoonify_s_d.pt', 64],
            'cartoon3-d': ['vtoonify_d_cartoon/vtoonify_s_d.pt', 153],
            'cartoon4': ['vtoonify_d_cartoon/vtoonify_s299_d0.5.pt', 299],
            'cartoon4-d': ['vtoonify_d_cartoon/vtoonify_s_d.pt', 299],
            'cartoon5-d': ['vtoonify_d_cartoon/vtoonify_s_d.pt', 8],
            'comic1-d': ['vtoonify_d_comic/vtoonify_s_d.pt', 28],
            'comic2-d': ['vtoonify_d_comic/vtoonify_s_d.pt', 18],
            'arcane1': ['vtoonify_d_arcane/vtoonify_s000_d0.5.pt', 0],
            'arcane1-d': ['vtoonify_d_arcane/vtoonify_s_d.pt', 0],
            'arcane2': ['vtoonify_d_arcane/vtoonify_s077_d0.5.pt', 77],
            'arcane2-d': ['vtoonify_d_arcane/vtoonify_s_d.pt', 77],
            'caricature1': ['vtoonify_d_caricature/vtoonify_s039_d0.5.pt', 39],
            'caricature2': ['vtoonify_d_caricature/vtoonify_s068_d0.5.pt', 68],
            'pixar': ['vtoonify_d_pixar/vtoonify_s052_d0.5.pt', 52],
            'pixar-d': ['vtoonify_d_pixar/vtoonify_s_d.pt', 52],
            'illustration1-d': ['vtoonify_d_illustration/vtoonify_s054_d_c.pt', 54],
            'illustration2-d': ['vtoonify_d_illustration/vtoonify_s004_d_c.pt', 4],
            'illustration3-d': ['vtoonify_d_illustration/vtoonify_s009_d_c.pt', 9],
            'illustration4-d': ['vtoonify_d_illustration/vtoonify_s043_d_c.pt', 43],
            'illustration5-d': ['vtoonify_d_illustration/vtoonify_s086_d_c.pt', 86],
        }
        
        self.landmarkpredictor = self._create_dlib_landmark_model()
        self.parsingpredictor = self._create_parsing_model()
        self.pspencoder = self._load_encoder()    
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
            ])
        
        self.vtoonify, self.exstyle = self._load_default_model()
        self.color_transfer = False
        self.style_name = 'cartoon1'
        self.video_limit_cpu = 100
        self.video_limit_gpu = 300
        
    @staticmethod
    def _create_dlib_landmark_model():
        return dlib.shape_predictor(huggingface_hub.hf_hub_download(MODEL_REPO,
                                                    'models/shape_predictor_68_face_landmarks.dat'))
    
    def _create_parsing_model(self):
        parsingpredictor = BiSeNet(n_classes=19)
        parsingpredictor.load_state_dict(torch.load(huggingface_hub.hf_hub_download(MODEL_REPO, 'models/faceparsing.pth'),
                                                    map_location=lambda storage, loc: storage))
        parsingpredictor.to(self.device).eval()
        return parsingpredictor
    
    def _load_encoder(self) -> nn.Module:
        style_encoder_path = huggingface_hub.hf_hub_download(MODEL_REPO,'models/encoder.pt')
        return load_psp_standalone(style_encoder_path, self.device)
    
    def _load_default_model(self) -> tuple[torch.Tensor, str]:
        vtoonify = VToonify(backbone = 'dualstylegan')
        vtoonify.load_state_dict(torch.load(huggingface_hub.hf_hub_download(MODEL_REPO,
                                            'models/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt'), 
                                            map_location=lambda storage, loc: storage)['g_ema'])
        vtoonify.to(self.device)
        tmp = np.load(huggingface_hub.hf_hub_download(MODEL_REPO,'models/vtoonify_d_cartoon/exstyle_code.npy'), allow_pickle=True).item()
        exstyle = torch.tensor(tmp[list(tmp.keys())[26]]).to(self.device)
        with torch.no_grad():  
            exstyle = vtoonify.zplus2wplus(exstyle)
        return vtoonify, exstyle
    
    def load_model(self, style_type: str) -> tuple[torch.Tensor, str]:
        if 'illustration' in style_type:
            self.color_transfer = True
        else:
            self.color_transfer = False
        if style_type not in self.style_types.keys():
            return None, 'Oops, wrong Style Type. Please select a valid model.'
        self.style_name = style_type
        model_path, ind = self.style_types[style_type]
        style_path = os.path.join('models',os.path.dirname(model_path),'exstyle_code.npy')
        self.vtoonify.load_state_dict(torch.load(huggingface_hub.hf_hub_download(MODEL_REPO,'models/'+model_path), 
                                            map_location=lambda storage, loc: storage)['g_ema'])
        tmp = np.load(huggingface_hub.hf_hub_download(MODEL_REPO, style_path), allow_pickle=True).item()
        exstyle = torch.tensor(tmp[list(tmp.keys())[ind]]).to(self.device)
        with torch.no_grad():  
            exstyle = self.vtoonify.zplus2wplus(exstyle)
        return exstyle, 'Model of %s loaded.'%(style_type)
    
    def detect_and_align(self, frame, top, bottom, left, right, return_para=False):
        message = 'Error: no face detected! Please retry or change the photo.'
        paras = get_video_crop_parameter(frame, self.landmarkpredictor, [left, right, top, bottom])
        instyle = None
        h, w, scale = 0, 0, 0
        if paras is not None:
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
            kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            with torch.no_grad():
                I = align_face(frame, self.landmarkpredictor)
                if I is not None:
                    I = self.transform(I).unsqueeze(dim=0).to(self.device)
                    instyle = self.pspencoder(I)
                    instyle = self.vtoonify.zplus2wplus(instyle)
                    message = 'Successfully rescale the frame to (%d, %d)'%(bottom-top, right-left)
                else:
                    frame = np.zeros((256,256,3), np.uint8)
        else:
            frame = np.zeros((256,256,3), np.uint8)
        if return_para:
            return frame, instyle, message, w, h, top, bottom, left, right, scale
        return frame, instyle, message
    
    #@torch.inference_mode()
    def detect_and_align_image(self, image: str, top: int, bottom: int, left: int, right: int
                              ) -> tuple[np.ndarray, torch.Tensor, str]:
        if image is None:
            return np.zeros((256,256,3), np.uint8), None, 'Error: fail to load empty file.'
        frame = cv2.imread(image)
        if frame is None:
            return np.zeros((256,256,3), np.uint8), None, 'Error: fail to load the image.'       
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return self.detect_and_align(frame, top, bottom, left, right)
    
    def detect_and_align_video(self, video: str, top: int, bottom: int, left: int, right: int
                              ) -> tuple[np.ndarray, torch.Tensor, str]:
        if video is None:
            return np.zeros((256,256,3), np.uint8), None, 'Error: fail to load empty file.'
        video_cap = cv2.VideoCapture(video)
        if video_cap.get(7) == 0:
            video_cap.release()
            return np.zeros((256,256,3), np.uint8), torch.zeros(1,18,512).to(self.device), 'Error: fail to load the video.'
        success, frame = video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_cap.release()
        return self.detect_and_align(frame, top, bottom, left, right)
    
    def detect_and_align_full_video(self, video: str, top: int, bottom: int, left: int, right: int) -> tuple[str, torch.Tensor, str]:
        message = 'Error: no face detected! Please retry or change the video.'
        instyle = None
        if video is None:
            return 'default.mp4', instyle, 'Error: fail to load empty file.'
        video_cap = cv2.VideoCapture(video)
        if video_cap.get(7) == 0:
            video_cap.release()
            return 'default.mp4', instyle, 'Error: fail to load the video.'    
        num = min(self.video_limit_gpu, int(video_cap.get(7)))
        if self.device == 'cpu':
            num = min(self.video_limit_cpu, num)
        success, frame = video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, instyle, message, w, h, top, bottom, left, right, scale = self.detect_and_align(frame, top, bottom, left, right, True)
        if instyle is None:
            return 'default.mp4', instyle, message    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter('input.mp4', fourcc, video_cap.get(5), (int(right-left), int(bottom-top)))
        videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
        for i in range(num-1):
            success, frame = video_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        videoWriter.release()
        video_cap.release()

        return 'input.mp4', instyle, 'Successfully rescale the video to (%d, %d)'%(bottom-top, right-left)
    
    def image_toonify(self, aligned_face: np.ndarray, instyle: torch.Tensor, exstyle: torch.Tensor, style_degree: float) -> tuple[np.ndarray, str]:
        if instyle is None or aligned_face is None:
            return np.zeros((256,256,3), np.uint8), 'Opps, something wrong with the input. Please go to Step 2 and Rescale Image/First Frame again.'
        if exstyle is None:
            return np.zeros((256,256,3), np.uint8), 'Opps, something wrong with the style type. Please go to Step 1 and load model again.'
        if exstyle is None:
            exstyle = self.exstyle
        with torch.no_grad():
            if self.color_transfer:
                s_w = exstyle
            else:
                s_w = instyle.clone()
                s_w[:,:7] = exstyle[:,:7]

            x = self.transform(aligned_face).unsqueeze(dim=0).to(self.device)
            x_p = F.interpolate(self.parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                scale_factor=0.5, recompute_scale_factor=False).detach()
            inputs = torch.cat((x, x_p/16.), dim=1)
            y_tilde = self.vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = style_degree)        
            y_tilde = torch.clamp(y_tilde, -1, 1)
        print('*** Toonify %dx%d image'%(y_tilde.shape[2], y_tilde.shape[3]))
        return ((y_tilde[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8), 'Successfully toonify the image with style of %s'%(self.style_name)
    
    def video_tooniy(self, aligned_video: str, instyle: torch.Tensor, exstyle: torch.Tensor, style_degree: float) -> tuple[str, str]:
        if aligned_video is None:
            return 'default.mp4', 'Opps, something wrong with the input. Please go to Step 2 and Rescale Video again.'         
        video_cap = cv2.VideoCapture(aligned_video)
        if instyle is None or aligned_video is None or video_cap.get(7) == 0:
            video_cap.release()
            return 'default.mp4', 'Opps, something wrong with the input. Please go to Step 2 and Rescale Video again.'
        if exstyle is None:
            return 'default.mp4', 'Opps, something wrong with the style type. Please go to Step 1 and load model again.'
        num = min(self.video_limit_gpu, int(video_cap.get(7)))
        if self.device == 'cpu':
            num = min(self.video_limit_cpu, num)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter('output.mp4', fourcc, 
                                      video_cap.get(5), (int(video_cap.get(3)*4),
                                      int(video_cap.get(4)*4)))        

        batch_frames = []
        if video_cap.get(3) != 0:
            if self.device == 'cpu':
                batch_size = max(1, int(4 * 256* 256/ video_cap.get(3) / video_cap.get(4)))
            else:
                batch_size = min(max(1, int(4 * 400 * 360/ video_cap.get(3) / video_cap.get(4))), 4)
        else:
            batch_size = 1
        print('*** Toonify using batch size of %d on %dx%d video of %d frames'%(batch_size, int(video_cap.get(3)*4), int(video_cap.get(4)*4), num))
        with torch.no_grad():
            if self.color_transfer:
                s_w = exstyle
            else:
                s_w = instyle.clone()
                s_w[:,:7] = exstyle[:,:7]
            for i in range(num):
                success, frame = video_cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_frames += [self.transform(frame).unsqueeze(dim=0).to(self.device)]
                if len(batch_frames) == batch_size or (i+1) == num:
                    x = torch.cat(batch_frames, dim=0)
                    batch_frames = []
                    with torch.no_grad():
                        x_p = F.interpolate(self.parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                            scale_factor=0.5, recompute_scale_factor=False).detach()
                        inputs = torch.cat((x, x_p/16.), dim=1)
                        y_tilde = self.vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), style_degree)       
                        y_tilde = torch.clamp(y_tilde, -1, 1)
                    for k in range(y_tilde.size(0)):
                        videoWriter.write(tensor2cv2(y_tilde[k].cpu()))
                    gc.collect()

        videoWriter.release()
        video_cap.release()
        return 'output.mp4', 'Successfully toonify video of %d frames with style of %s'%(num, self.style_name)

