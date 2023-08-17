import json
from torch.utils.data import Dataset
import numpy as np
import random
import time
from dataset.augment import *
import nibabel as nib

class MedKLIP_Dataset(Dataset):
    def __init__(self, csv_path, np_path):
        self.ann = json.load(open(csv_path,'r'))
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)

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
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        for modal in modal_dic:
            data=nib.load(self.ann[fid][modal])
            img_data=data.get_fdata()
            if img_data.ndim>3:
                img_data=img_data[:,:,:,self.ann[fid]['component']]
            #img_data=self.normalize(img_data)
            #image=downscale(img_data,[224,224,24])
            image = nnUNet_resample_and_normalize(img_data,[224,224,24])
            image = image.transpose([2,1,0])
            image=image[np.newaxis,:]
            image_sum.append(image)

        return {
            "image": image_sum,
            "label": labels,
            "fid":fid
            }
      
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1 # [-1]
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


class MedKLIP_Dataset_randomchoice(Dataset):
    def __init__(self, csv_path, np_path):
        self.ann = json.load(open(csv_path,'r'))
        self.fid_list_origin = list(self.ann)
        self.fid_list=self.randomchoice(self.fid_list_origin)
        self.rad_graph_results = np.load(np_path)

    def randomchoice(self,fid_list):
        random_fid=[]
        seed=int(time.time())
        random.seed(seed)
        for i in range(len(fid_list)):
            fid=random.choice(fid_list)
            random_fid.append(fid)
        return random_fid

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
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        for modal in modal_dic:
            data=nib.load(self.ann[fid][modal])
            img_data=data.get_fdata()
            if img_data.ndim>3:
                img_data=img_data[:,:,:,self.ann[fid]['component']]
            #img_data=self.normalize(img_data)
            #image=downscale(img_data,[224,224,24])
            image = nnUNet_resample_and_normalize(img_data,[224,224,24])
            image = image.transpose([2,1,0])
            image=image[np.newaxis,:]
            image_sum.append(image)

        return {
            "image": image_sum,
            "label": labels,
            "fid":fid
            }
      
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1 # [-1]
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


class MedKLIP_Vis_Dataset(Dataset):
    def __init__(self, csv_path, np_path,exclude_fid=[]):
        self.ann = json.load(open(csv_path,'r'))
        for fid in exclude_fid:
            del self.ann[fid]
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)

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
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        image_unnorm_sum = []
        for modal in modal_dic:
            data=nib.load(self.ann[fid][modal])
            img_data=data.get_fdata()
            if img_data.ndim>3:
                img_data=img_data[:,:,:,self.ann[fid]['component']]
            #img_data=self.normalize(img_data)
            #image=downscale(img_data,[224,224,24])
            image = nnUNet_resample_and_normalize(img_data,[224,224,24],resize_mode="normal")
            image_unnorm = nnUNet_resample_and_normalize(img_data,[224,224,24],normalize=False,resize_mode="normal")
            image = image.transpose([2,1,0])
            image_unnorm = image_unnorm.transpose([2,1,0])
            image = image[np.newaxis,:]
            image_unnorm = image_unnorm[np.newaxis, :]
            image_sum.append(image)
            image_unnorm_sum.append(image_unnorm)

        return {
            "image": image_sum,
            "image_unnorm": image_unnorm_sum,
            "label": labels,
            "fid":fid
            }
      
    def triplet_extraction(self, class_label):
        exist_labels = np.zeros(class_label.shape[-1]) -1 # [-1]
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

