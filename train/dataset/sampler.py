from torch.utils.data import Sampler
import numpy as np
import random
import math
import time

class UniformSampler(Sampler):
    def __init__(self,dataset,batch_size,batch_clas_num=8):
        self.fid_list = dataset.fid_list
        self.fid_info = dataset.ann
        self.labels = dataset.rad_graph_results
        self.n = len(dataset)
        self.class_id = [i for i in range(self.labels.shape[-1])]
        self.batch_size = batch_size
        self.batch_clas_num = batch_clas_num
        self.dis_dic = self.generate_dis_dic()
    
    def generate_dis_dic(self):
        dis_dic = {i:[] for i in self.class_id}
        for idx,fid in enumerate(self.fid_list):
            class_label = self.labels[self.fid_info[fid]["labels_id"],:,:] # ana_num, dis_num
            dis_id = np.where(class_label.sum(axis=0)>0)[0]
            for i in dis_id:
                dis_dic[i].append(idx)
        return dis_dic

    def __iter__(self):
        idxs = []
        group = math.ceil(self.n/self.batch_size)
        # 每个类别分别抽取batch_size // n个
        for _ in range(group):
            # 抽取batch_cls_num个类别
            random.seed(time.time())
            cur_cls_id = random.sample(self.class_id,self.batch_clas_num)
            each_cls_num = int(self.batch_size//self.batch_clas_num)
            for idx in cur_cls_id:
                random.seed(time.time())
                idxs += random.sample(self.dis_dic[idx],each_cls_num)
        return iter(idxs)
    
    def __len__(self):
        return math.ceil(self.n/self.batch_size)*self.batch_size