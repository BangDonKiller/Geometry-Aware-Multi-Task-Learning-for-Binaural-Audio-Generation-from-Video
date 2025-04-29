import numpy as np
import torch
from Models.backbone_model import *
from Models.geometry_model import *
from Models.spatial_model import *
from Models.rir_model import *
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from params import *


class model(torch.nn.Module):

    def __init__(self, nets):
        super(model, self).__init__()
        #initialize model
        self.visual_net, self.spatial_net, self.audio_net, self.fusion_net, self.generator = nets
        self.device = device

    def forward(self, data, mode='train'):
        backbone_nets = (self.audio_net, self.fusion_net)
        model_backbone = modelBackbone(backbone_nets)
        model_spatial = modelSpatial(self.spatial_net)
        model_rir = modelRir(self.generator)

         # visual forward
        visual_input = data['frame'].to(device)
        visual_feature, visual_feature_flat = self.visual_net.forward(visual_input)
                
        # backbone forward
        output_backbone = model_backbone.forward(data, visual_feature, visual_feature_flat)
        
        # geometric consistency forward 
        second_visual_input = data['second_frame'].to(device)
        second_visual_feature, second_visual_feature_flat = self.visual_net.forward(second_visual_input)
        
        # rir forward
        # rir_output = model_rir.forward(rir_input)
        
                
        # spatial coherence forward
        if mode != 'test':
            output_spatial = model_spatial(data, visual_feature)
            return {"visual_feature" : visual_feature, "second_visual_feature" : second_visual_feature, **output_backbone, **output_spatial}
        
        
        return {"visual_feature" : visual_feature, "second_visual_feature" : second_visual_feature, **output_backbone}
        
    

        

