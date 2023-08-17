# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
#from sklearn.metrics import log_loss
import json
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel
from models import resnet,densenet

'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''


class MedKLIP(nn.Module):

    def __init__(self, config, mode='train'):
        super(MedKLIP, self).__init__()

        self.mode = mode
        self.d_model = config['d_model']
        
        self.cl_fc = nn.Linear(config['out_feature'],768)
        self.excluded_disease = ['normal']
        self.disease_name = json.load(open(config['disease_order'],'r'))
        self.cl_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease ]   
        
        ''' visual backbone'''
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        # resnet = self._get_res_basemodel(config['res_base_model'])
        # num_ftrs = int(resnet.fc./2)
        # self.res_features = nn.Sequential(*list(resnet.children())[:-3])

        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , self.decoder_norm,
                                  return_intermediate=False)

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        self.apply(self._init_weights)

        self.config = config
    
    def forward(self, image_feature, text_features, return_ws=False):
        # image_features 4, b,2352,768
        
        text_features = text_features.transpose(0,1)
        # ana_features = ana_features[0,:,:]

        img_feature = image_feature.transpose(0,1) # n b d
        # 14 8 768 2352 8 768
        # img_feature = self.decoder_norm(img_feature)
        # text_features = self.decoder_norm(text_features)
        feature,ws = self.decoder(text_features, img_feature, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        ws_mean=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4

        # no attention
        out = self.dropout_feas(feature)
       
        # print("out",out.shape)
        
        x = self.classifier(out).transpose(0,1) #B query Atributes  8,16,2

        if return_ws:
            return ws_mean
        else:
            return x

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()