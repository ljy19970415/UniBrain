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
from models import resnet

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

        # focal_loss_weight
        # self.class_weight = json.load(open(config['class_weight'],'r'))
        # LA
        # disease_order=json.load(open(config['disease_order'],'r'))
        # class_p = json.load(open(config['class_p'],'r'))
        # self.class_p = torch.tensor(config["la_alpha"])*torch.log(torch.tensor([[class_p[i][0]/class_p[i][1]] for i in disease_order]))
        self.config = config
    
    def forward(self, image_feature, text_features, ana_features=None, sample_index=None, celoss=False):
        # image_features 4, b,2352,768
        B = image_feature.shape[0]
        device = image_feature.device
        text_features = text_features.transpose(0,1)
        ana_features = ana_features[0,:,:] if ana_features is not None else None

        img_feature = image_feature.transpose(0,1) # n b d
        # 14 8 768 2352 8 768
        # img_feature = self.decoder_norm(img_feature)
        # text_features = self.decoder_norm(text_features)
        feature,ws = self.decoder(text_features, img_feature, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        # ws_mean=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4

        # no attention
        out = self.dropout_feas(feature)

        if self.config['no_cl'] == False:
            anatomy_query = ana_features[sample_index,:] # batch, 4 , dim
            # [Q,B,A]
            # smaple_index.shape torch.Size([8, 14, 8]) 
            # ana_features torch.Size([65, 768])
            # anatomy_query.shape torch.Size([8, 14, 8, 768])
            ll = out.transpose(0,1) # B Q A
            Q = ll.shape[1]
            ll = ll.reshape(ll.shape[0]*ll.shape[1],-1)
            ll = self.cl_fc(ll)
            ll = ll.unsqueeze(dim =-1)
            #ll = ll.reshape(B,Q,-1)
            anatomy_query = anatomy_query.reshape(B*Q,8,768)
            ll = torch.bmm(anatomy_query, ll).squeeze()  # B Q 4
            cl_labels = torch.zeros((ll.shape[0])).to(device)
            #if exclude_class == True:
            cl_labels = cl_labels.reshape(B,Q)
            cl_labels = cl_labels[:,self.cl_class_dim]
            cl_labels = cl_labels.reshape(-1)
            ll = ll.reshape(B,Q,-1)
            ll = ll[:,self.cl_class_dim,:]
            ll = ll.reshape(B*(len(self.cl_class_dim)),-1)
        
        x = self.classifier(out).transpose(0,1) #B query Atributes  8,16,2

        if celoss:
            return x.squeeze()

        logits = x.reshape(-1, x.shape[-1])
        if self.config["la"]:
            class_p = class_p.unsqueeze(0).repeat(B,1,1)
            class_p = class_p.reshape(-1,class_p.shape[-1])
            logits = logits + class_p
        logits = torch.sigmoid(logits)

        if self.config['no_cl'] == False:
            return logits, ll, cl_labels
        else:
            return logits

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