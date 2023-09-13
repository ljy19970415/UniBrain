
import cv2
import imageio

import argparse
import os
import yaml as yaml
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import io

from models.model import UniBrain as UniBrain

from dataset.dataset import MRI_Vis_Dataset as MRI_Dataset
from dataset.augment import *
from models.tokenization_bert import BertTokenizer
from transformers import AutoModel
from models.imageEncoder import ModelRes, ModelDense
from models.before_fuse import *

import os

from einops import rearrange

import imageio

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = False

    return model

def get_text_features(model,text_list,tokenizer,device,max_length):
    # text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    target_tokenizer = tokenizer(list(text_list), padding='max_length', truncation=True, max_length=max_length,return_tensors="pt").to(device)
    # text_features = model.encode_text(text_token)
    text_features = model(input_ids = target_tokenizer['input_ids'],attention_mask = target_tokenizer['attention_mask'])#(**encoded_inputs)
    text_features = text_features.last_hidden_state[:,0,:]
    # text_features = F.normalize(text_features, dim=-1)
    return text_features

def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def test(args,config,keep_largest_mask = True):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    file_key = args.mode+'_file'

    fid_ex=set()
    # for r,ds,filename in os.walk(args.output_dir):
    #     for d in ds:
    #         if d.startswith("a"):
    #             fid_ex.add(d)

    test_dataset =  MRI_Dataset(config[file_key],config['label_file'],fid_ex) 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=1,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
        )
    
    print("Creating book")
    all_target_class = json.load(open(config['disease_order'],'r'))
    target_class = all_target_class[:13].copy()
    # target_class=json.load(open(config['disease_order'],'r'))
    if "exclude_class" in config and config["exclude_class"]:
        keep_class_dim = [target_class.index(i) for i in target_class if i not in config["exclude_classes"] ]
        target_class = [target_class[i] for i in keep_class_dim]
        keep_class_dim = [all_target_class.index(i) for i in all_target_class if i not in config["exclude_classes"] ]
        all_target_class = [target_class[i] for i in keep_class_dim]
    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    dis_fid = json.load(open(config['dis_high_prob'],'r'))
    disease_book = [json_book[i] for i in json_order]
    # disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)

    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    text_encoder = _get_bert_basemodel(config['text_encoder']).to(device)
    text_features = get_text_features(text_encoder,disease_book,tokenizer,device,max_length=256)

    if config['model_type']== 'resnet':
        image_encoder =ModelRes(config).to(device)
    elif config['model_type'] == 'densenet':
        image_encoder = ModelDense(config).to(device)

    fuseModule = beforeFuse(config).to(device) # before fusion
    
    print("Creating model")

    model = UniBrain(config)

    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)

    print('Load model from checkpoint:',args.model_path)
    checkpoint = torch.load(args.model_path,map_location='cpu') 
    state_dict = checkpoint['model']          
    model.load_state_dict(state_dict)

    # for name, param in model.named_parameters():
    #     if "classifier" in name:
    #         print("init",name,param.shape)
    #         h = rearrange(param,'c d l -> c (d l)')
    #         print(torch.sum(h,dim=1))
    
    image_encoder.load_state_dict(checkpoint['image_encoder'])
    image_encoder = nn.DataParallel(image_encoder, device_ids = [i for i in range(torch.cuda.device_count())])
    image_encoder = image_encoder.to(device)
    # image_encoder.load_state_dict(checkpoint['image_encoder'])

    fuseModule.load_state_dict(checkpoint['fuseModule'])
    # fuseModule = nn.DataParallel(fuseModule, device_ids = [i for i in range(torch.cuda.device_count())])
    # fuseModule = fuseModule.to(device)

    # initialize the ground truth and output tensor
    fids = []
    
    print("Start visualizing")
    model.eval()

    for i, sample in enumerate(test_dataloader):
        images = sample['image']  # [(b,24,224,224),(b,x,y,z),(b,x,y,z),(b,x,y,z)]
        image_unnorm = sample['image_unnorm']
        labels=sample['label'][:,:].float() # (b,c)
        fids=sample['fid']
        
        B = labels.shape[0]
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        
        cur_text_features = text_features.unsqueeze(0).repeat(B,1,1)

        image_features = [] # image_features 4 b n d, image_features_pool 4 b d
        image_features_pool = []
        for idx,cur_image in enumerate(images):
            image_feature,image_feature_pool = image_encoder(cur_image) 
            image_features.append(image_feature)
            image_features_pool.append(image_feature_pool)
        
        fuse_image_feature = fuseModule(image_features)
        
        # input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            ws = model(fuse_image_feature, cur_text_features, return_ws=True) #[(b,c,patch_num),(),(),()]
            # resize attentionmap use nnUNet_resample_and_normalize(data, new_shape, normalize=False)
            for index in range(labels.shape[0]):  # sample fid one by one
                fid = fids[index]
                # print("fid",fid)
                for class_index in range(len(json_order)):
                    if labels[index,class_index] == 1 and fid in dis_fid[json_order[class_index]]:
                        # print("dis",json_order[class_index])
                        # img_size = (img_file.shape[0],img_file.shape[1])
                        atten_map = rearrange(ws[index,class_index,:],' (z w h) ->z w h',z=3,w=28,h=28)
                        atten_map = np.array(atten_map.cpu().numpy())
                        atten_map = nnUNet_resample_and_normalize(atten_map,[24,224,224],normalize=False)
                        atten_map = cv2.normalize(atten_map,None,0,255,cv2.NORM_MINMAX)
                        largest = 0
                        largest_mask = None
                        for z in range(24):
                            atten_map_z=np.asarray(atten_map[z,:,:]).astype(np.uint8)
                            atten_map_z =atten_map_z.flatten()
                            if np.sum(atten_map_z)>largest:
                                largest_mask = atten_map_z
                                largest = np.sum(atten_map_z)
                        largest_mask[np.argpartition(largest_mask,-100)[-100:]] = max(largest_mask)
                        largest_mask = rearrange(largest_mask,'(w h) ->w h',w=224,h=224)
                        save_heatmap_npy=os.path.join(args.output_dir,str(json_order[class_index]),str(fids[index]))
                        os.makedirs(save_heatmap_npy,exist_ok=True)
                        np.save(os.path.join(save_heatmap_npy,"heatmap.npy"),largest_mask)
                        largest_mask = cv2.applyColorMap(largest_mask,cv2.COLORMAP_JET) # (224,224,3)
                        largest_mask = cv2.normalize(largest_mask,None,0,255,cv2.NORM_MINMAX)
                        for modal_index in range(len(modal_dic)):
                            for z in range(24):
                                save_root=os.path.join(args.output_dir,str(json_order[class_index]),str(fids[index]),str(modal_dic[modal_index]))
                                os.makedirs(save_root,exist_ok=True)
                                save_mask_path = os.path.join(save_root,str(z)+'_mask.png')
                                save_img_path = os.path.join(save_root,str(z)+'_img.png')
                                atten_map_z=np.asarray(atten_map[z,:,:]).astype(np.uint8)
                                atten_map_z =atten_map_z.flatten()
                                atten_map_z[np.argpartition(atten_map_z,-100)[-100:]] = max(atten_map_z)
                                atten_map_z =rearrange(atten_map_z,'(w h) ->w h',w=224,h=224)
                                atten_map_z = cv2.applyColorMap(atten_map_z,cv2.COLORMAP_JET) # (224,224,3)
                                atten_map_z = cv2.normalize(atten_map_z,None,0,255,cv2.NORM_MINMAX)
                                img_array = np.asarray(images[modal_index][index,0,z,:,:]).astype(np.uint8)
                                imageio.imwrite(save_img_path,images[modal_index][index,0,z,:,:])
                                # np.save(os.path.join(save_root,str(z)+'_img.npy'),img_array)
                                img_array = cv2.normalize(img_array,None,0,255,cv2.NORM_MINMAX)
                                img_array = np.stack((img_array,)*3,axis=-1)  #channel 1 -> 3
                                # io.imsave(save_img_path,img_array)
                                if keep_largest_mask:
                                    atten_map_img=cv2.addWeighted(img_array,0.5, largest_mask,0.5, 0)
                                    # atten_map_img = largest_mask
                                else:
                                    atten_map_img=cv2.addWeighted(img_array,0.5, atten_map_z,0.5, 0)
                                io.imsave(save_mask_path,cv2.cvtColor(atten_map_img,cv2.COLOR_BGR2RGB))
                                # imageio.imwrite(save_mask_path,atten_map_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config_visualize_jiayu_show.yaml')
    parser.add_argument('--model_path', default='./pretrained_weights/best_val.pth')
    parser.add_argument('--output_dir', default='./grounding_output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0,1,2,3', help='gpu')
    parser.add_argument('--mode', type=str,default='test')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != '-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    test(args, config, keep_largest_mask=True)