import json
from torch.utils.data import Dataset
import numpy as np
import random
from .augment import *
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *

class MedKLIP_Dataset(Dataset):
    def __init__(self, input_case_dict, np_path):

        # T1WI 1 T2WI 9 T2FLAIR 12 DWI 18
        # input_case_dict = {
        # "patient1":{
        # '18':'example/site1_547/DWI.nii.gz',
        # '1':'example/site1_547/T1WI.nii.gz',
        # '9':"example/site1_547/T2WI.nii.gz",
        # '12':'example/site1_547/T2FLAIR.nii.gz'}
        # }

        # self.ann = json.load(open(csv_path,'r'))
        self.ann = input_case_dict
        self.fid_list = list(self.ann)
        self.rad_graph_results = np.load(np_path)
        modal_id = json.load(open('Brain_MRI/configs/modal_id.json','r'))
        self.modal_id = {modal_id[i]:i for i in modal_id}

    def normalize(self,image):
        MIN_BOUND, MAX_BOUND = 0,1000
        image = np.clip(image, MIN_BOUND, MAX_BOUND)
        image = 2 * (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - 1
        return image
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        class_label = self.rad_graph_results[0,:,:] # (51, 75)
        labels = np.zeros(class_label.shape[-1]) -1
        labels, index_list = self.triplet_extraction(class_label)
        modal_dic=["DWI","T1WI","T2WI","T2FLAIR"]
        image_sum=[]
        for modal in modal_dic:
            # print(modal)
            if self.modal_id[modal] not in self.ann[fid]:
                image = np.zeros((224,224,24))
            else:
                data=nib.load(self.ann[fid][self.modal_id[modal]])
                img_data=data.get_fdata()
                # print("img_data.shape",img_data.shape)
                if img_data.ndim>3:
                    img_data=img_data[:,:,:,1]
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
