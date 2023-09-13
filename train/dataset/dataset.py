import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random
from dataset.augment import *
import nibabel as nib
import os

from augmentation.data_trans import *

def nib_load(file_name, component):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    if data.ndim>3:
        data=data[:,:,:,component]
    proxy.uncache()
    return data

class MRI_Dataset(Dataset):
    def __init__(self, csv_path, np_path, report_observe, mode = 'train', augmentation=False, only_global=False,mask_modal=""):
        self.ann = json.load(open(csv_path,'r'))
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)
        self.report = np.load(report_observe,allow_pickle='True').item()
        self.augmentation = augmentation
        self.only_global = only_global
        self.mask_modal = mask_modal

    def normalize(self,image):
        MIN_BOUND, MAX_BOUND = 0,1000
        image = np.clip(image, MIN_BOUND, MAX_BOUND)
        image = 2 * (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - 1
        return image
    
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        class_label = self.rad_graph_results[self.ann[fid]["labels_id"],:,:] # (51, 75)
        labels = np.zeros(class_label.shape[-1]) -1
        labels, index_list = self.triplet_extraction(class_label)
        index_list = np.array(index_list)
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        entity = []
        report_entity = []
        if self.augmentation:
            images = []
            for modal in modal_dic:
                data = np.array(nib_load(self.ann[fid][modal], self.ann[fid]['component']), dtype='float32', order='C')
                # image=self.transform(img_data)
                image = nnUNet_resample(data,[224,224,24],is_seg=False)
                images.append(image)
            images = np.stack(images, -1)
            mask = images.sum(-1) > 0
            for k in range(4):
                x = images[..., k]
                y = x[mask]
                mn = y.mean()
                std = y.std()
                x[mask] = (x[mask] - mn) / (std + 1e-8)
                images[..., k] = x
            images = transform(images)
            image_sum = [images[...,k].transpose([2,1,0])[np.newaxis,:] for k in range(len(modal_dic))]
        else:
            for modal in modal_dic:
                data=nib.load(self.ann[fid][modal])
                img_data=data.get_fdata()
                if img_data.ndim>3:
                    img_data=img_data[:,:,:,self.ann[fid]['component']]
                image = nnUNet_resample_and_normalize(img_data,[224,224,24],is_seg=False)
                image = image.transpose([2,1,0])
                image=image[np.newaxis,:]
                image_sum.append(image)
        if self.mask_modal != "":
            modal_idx = modal_dic.index(self.mask_modal)
            image_sum[modal_idx] = np.zeros(image_sum[modal_idx].shape)
        if not self.only_global:
            for modal in modal_dic:
                # entity.append('[SEP]'.join(self.report[fid]['fuse']))
                entity.append('[SEP]'.join(self.report[fid][modal]))
                # if len(self.report[fid][modal]) == 1 and (self.report[fid][modal][0] == "isointensity" or self.report[fid][modal][0] == 'unspecified'):
                #     report_entity.append(modal+' '+self.report[fid][modal][0])
                # else:
                #     report_entity.append('[SEP]'.join(self.report[fid][modal]))
        if 'fuse' in self.report[fid]:
            entity.append('[SEP]'.join(self.report[fid]['fuse']))
        
        # img_path = self.img_path_list[index]
        class_label = self.rad_graph_results[self.ann[fid]["labels_id"],:,:] # (51, 75)
        labels = np.zeros(class_label.shape[-1]) -1
        labels, index_list = self.triplet_extraction(class_label)
        index_list = np.array(index_list)

        # report = '[SEP]'.join(report_entity)

        return {
            "image": image_sum,
            "label": labels,
            'index': index_list,
            'entity': entity,
            # 'report':report,
            "fid":fid
            }
    
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1
        position_list = []
        for i in range(class_label.shape[1]):
            temp_list = []
            ### extract the exist label for each entity and maintain -1 if not mentioned. ###
            if 0 in class_label[:,i]:
                exist_labels[i] = 0
                
            if 1 in class_label[:,i]:
                exist_labels[i] = 1
                ### if the entity exists try to get its position.### 
                ### Note that, the contrastive loss will only be caculated on exist entity as it is meaningless to predict their position for the non-exist entities###
                temp_list.append(random.choice(np.where(class_label[:,i] == 1)[0]))
                try:
                    temp_list = temp_list + random.sample(np.where(class_label != 1)[0].tolist(),7)
                except:
                    print('fatal error')
            if temp_list == []:
                temp_list = temp_list +random.sample(np.where(class_label != 1)[0].tolist(),8)
            position_list.append(temp_list)
        return exist_labels, position_list
    
    def __len__(self):
        return len(self.fid_list)

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

