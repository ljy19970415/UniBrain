import yaml as yaml
import numpy as np
import random
import json
import os
from transformers import AutoModel
from batchgenerators.utilities.file_and_folder_operations import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.diagnosis.model_MedKLIP_before_fuse import MedKLIP as MedKLIP 
from .models.diagnosis.tokenization_bert import BertTokenizer
from .models.diagnosis.imageEncoder import ModelRes
from .models.diagnosis.before_fuse import *
from .models.diagnosis.dataset import MedKLIP_Dataset
from .models.diagnosis.utils import *

import sys

if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

from .models.utilities.llm_metric import *
from .models.run.load_pretrained_weights import *
from .models.utilities.nd_softmax import *

class RatiocinationSdk():
    def __init__(self, gpu_id, inference_cfg):
        self.diagnosis_modal = DiagnosisModel(gpu_id,inference_cfg)

    def diagRG(self,input_case):

        result = []
        for item in input_case:
            input_case_list_diag = []
            input_case_list_diag.append({j['aux']:j['data'] for j in item})
            results_diag = self.diagnosis_modal.diagnosis(input_case_list_diag)

            result.append({'diagnosis':results_diag[0]})
        
        return result
        
class DiagnosisModel():
    def __init__(self, gpu_id, inference_cfg):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        config = yaml.load(open(inference_cfg, 'r'), Loader=yaml.Loader)
        self.all_target_class = json.load(open(config['disease_order'],'r'))
        self.json_book = json.load(open(config['disease_book'],'r'))
        self.json_order=json.load(open(config['disease_order'],'r'))
        self.disease_book = [self.json_book[i] for i in self.json_order]

        self.tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
        self.text_encoder = _get_bert_basemodel(config['text_encoder']).to(device)
        self.text_features = get_text_features(self.text_encoder,self.disease_book,self.tokenizer,device,max_length=256)

        self.image_encoder = ModelRes(config).to(device)
        self.fuseModule = beforeFuse(config).to(device) # before fusion
    

        self.model = MedKLIP(config)

        self.model = nn.DataParallel(self.model, device_ids = [i for i in range(torch.cuda.device_count())])
        self.model = self.model.to(device)

        print('Load model from checkpoint:',config['pretrain_weight'])
        checkpoint = torch.load(config['pretrain_weight'], map_location='cpu') 
        state_dict = checkpoint['model']      
        self.model.load_state_dict(state_dict)
        
        self.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.image_encoder = nn.DataParallel(self.image_encoder, device_ids = [i for i in range(torch.cuda.device_count())])
        self.image_encoder = self.image_encoder.to(device)
        # image_encoder.load_state_dict(checkpoint['image_encoder'])

        self.fuseModule.load_state_dict(checkpoint['fuseModule'])
        self.fuseModule = nn.DataParallel(self.fuseModule, device_ids = [i for i in range(torch.cuda.device_count())])
        self.fuseModule = self.fuseModule.to(device)

        self.id_dis = json.load(open('Brain_MRI/configs/dis_order_id.json','r'))

        self.dis_id = { self.id_dis[i]:i for i in self.id_dis}

        self.model.eval()

    def diagnosis(self, input_case_list):

        input_case_dict = {}
        for i in range(len(input_case_list)):
            input_case_dict[str(i)] = input_case_list[i]
        
        test_dataset = MedKLIP_Dataset(input_case_dict, 'Brain_MRI/configs/label1.npy') 
        test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                sampler=None,
                shuffle=True,
                collate_fn=None,
                drop_last=False,
            ) 
        
        pred = torch.FloatTensor()
        fids = []

        for i, sample in enumerate(test_dataloader):
            images = sample['image']  # [(b,x,y,z),(b,x,y,z)]
            # labels = sample['label'].to(device)
            label = sample['label'][:,:].float()

            B = label.shape[0]

            cur_text_features = self.text_features.unsqueeze(0).repeat(B,1,1)

            image_features = [] # image_features 4 b n d, image_features_pool 4 b d
            image_features_pool = []
            for idx,cur_image in enumerate(images):
                image_feature,image_feature_pool = self.image_encoder(cur_image) 
                image_features.append(image_feature)
                image_features_pool.append(image_feature_pool)
            
            # before fuse
            fuse_image_feature = self.fuseModule(image_features)
            
            #input_image = image.to(device,non_blocking=True)  
            with torch.no_grad():
                pred_class = self.model(fuse_image_feature,cur_text_features, return_ws=False) #batch_size,num_class,1
                # input()
                pred_class = torch.sigmoid(pred_class.reshape(-1,1)).reshape(-1,len(self.all_target_class))
                pred = torch.cat((pred, pred_class.detach().cpu()), 0)
                fids += sample["fid"]

        threshs = json.load(open('Brain_MRI/configs/thresh.json','r'))
        threshs = list(map(lambda x:float(x), threshs))

        output_case_dict = {}
        pred = pred.cpu().numpy()
        for j in range(len(pred)):
            fid = fids[j]
            output_case_dict[fid]=[]
            for i in range(len(self.all_target_class)):
                if pred[j][i] >= threshs[i]:
                    # output_case_dict[fid].append(self.dis_id[self.all_target_class[i]])
                    output_case_dict[fid].append(self.all_target_class[i])
            output_case_dict[fid].sort()
        
        output_case_list = []

        for i in range(len(input_case_list)):
            output_case_list.append(output_case_dict[str(i)])
        
        if len(output_case_list) == 0:
            output_case_list = ["normal"]
        
        return output_case_list



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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_save_to_queue_seg(preprocess_fn, q, list_of_lists, list_of_segs, modals, output_files, transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            the_seg = list_of_segs[i] if list_of_segs is not None else None
            target_shape = None
            d, s, dct = preprocess_fn(l, the_seg, target_shape=target_shape)

            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            modal = modals[i]
            q.put((output_file, l, modal, (d, s, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded_seg(trainer, list_of_lists, list_of_segs, modals, output_files, num_processes=2):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    # classes = list(range(1, trainer.num_classes)) # 96

    # assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []

    for i in range(num_processes):
        the_segs = list_of_segs[i::num_processes] if list_of_segs is not None else None

        pr = Process(target=preprocess_save_to_queue_seg, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes], the_segs,
                                                            modals[i::num_processes],
                                                            output_files[i::num_processes], trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        q.close()
    
def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, transpose_forward):

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            the_ab_seg = list_of_ab_segs[i] if list_of_ab_segs is not None else None
            the_ana_seg = list_of_ana_segs[i] if list_of_ana_segs is not None else None

            target_shape = None

            if the_ab_seg is not None:
                d, s_ab, dct = preprocess_fn(l, the_ab_seg, target_shape=target_shape)
            else:
                s_ab = None
            
            if the_ana_seg is not None:
                d, s_ana, dct = preprocess_fn(l, the_ana_seg, target_shape=target_shape)
            else:
                s_ana = None

            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                # np.save(output_file[:-7] + ".npy", d)
                # d = output_file[:-7] + ".npy"
                print(l)
            r = list_of_reports[i] if list_of_reports is not None else None
            identi = case_identifiers[i]
            modal = modals[i]
            q.put((identi, modal, l, the_ab_seg, (r, d, s_ab, s_ana, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(trainer, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, num_processes=6):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    q = Queue(1)
    processes = []

    for i in range(num_processes):
        the_ab_segs = list_of_ab_segs[i::num_processes] if list_of_ab_segs is not None else None
        the_ana_segs = list_of_ana_segs[i::num_processes] if list_of_ana_segs is not None else None
        the_reports = list_of_reports[i::num_processes] if list_of_reports is not None else None
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes], the_ab_segs, the_ana_segs, the_reports,case_identifiers[i::num_processes],modals[i::num_processes],
                                                            trainer.plans['transpose_forward']))
        # pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
        #                                                     list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports,case_identifiers,modals,
        #                                                     trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        q.close()
