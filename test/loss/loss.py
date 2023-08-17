import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import json

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ASLLoss(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        class_p = json.load(open(config['asl_loss_weight'],'r'))
        dis_order = json.load(open(config['disease_order'],'r'))
        self.class_y_pos = np.expand_dims(np.array([class_p[i][0] for i in dis_order]),axis=0) # c,
        self.class_m_pos = np.expand_dims(np.array([class_p[i][1] for i in dis_order]),axis=0)
        self.class_y_neg = np.expand_dims(np.array([class_p[i][2] for i in dis_order]),axis=0)
        self.class_m_neg = np.expand_dims(np.array([class_p[i][3] for i in dis_order]),axis=0)
        self.c = len(dis_order)
    
    def forward(self, preds, gts, B):
        # preds b*class_num
        # gts b*class_num
        device = preds.device
        preds = preds.detach().cpu().numpy()
        #print("preds",preds[:self.c],gts[:self.c])
        class_y_pos = np.repeat(self.class_y_pos,B,axis=0).flatten() # b*c
        class_y_neg = np.repeat(self.class_y_neg,B,axis=0).flatten()
        class_m_pos = np.repeat(self.class_m_pos,B,axis=0).flatten()
        class_m_neg = np.repeat(self.class_m_neg,B,axis=0).flatten()
        #print("weight",class_y_pos[:self.c],class_m_pos[:self.c],class_y_neg[:self.c],class_m_neg[:self.c])
        preds_pos = preds+class_m_pos
        #print("+m",preds_pos[:self.c])
        preds_pos = np.array(list(map(lambda x:min(x,1), preds_pos)))
        #print("min",preds_pos[:self.c])
        preds_pos = np.power(1-preds_pos,class_y_pos)*np.log(preds_pos)
        #print("L+",preds_pos[:self.c])
        preds_pos = torch.tensor(preds_pos).to(device)

        preds_neg = preds-class_m_neg
        #print("-m",preds_neg[:self.c])
        preds_neg = np.array(list(map(lambda x:max(x,0), preds_neg)))
        #print("max",preds_neg[:self.c])
        preds_neg = np.power(preds_neg,class_y_neg)*np.log(1-preds_neg)
        #print("L-",preds_neg[:self.c])
        preds_neg = torch.tensor(preds_neg).to(device)

        ce_loss = -gts*preds_pos - (1-gts)*preds_neg

        #print("ce_loss",ce_loss[:self.c])
        #print(self.co)

        ce_loss = torch.mean(ce_loss)

        #print("ce_loss",ce_loss.item())
        
        return ce_loss

class chexzeroLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image_features, text_features, labels, criterion, mode="train"):
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        if mode == "val":
            return logits_per_image

        loss_img = criterion(logits_per_image, labels)
        loss_txt = criterion(logits_per_text, labels)
        loss = (loss_img + loss_txt)/2 # avg. img and txt loss

        return loss

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
    def forward(self, image_features, text_features, labels):
        # image_features 4 b d
        # text_features 4 b d
        # labels 4 b b
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        device = image_features[0].device
        # if self.world_size > 1:
        #     all_image_features, all_text_features = gather_features(
        #         image_features, text_features,
        #         self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        #     if self.local_loss:
        #         logits_per_image = logit_scale * image_features @ all_text_features.T
        #         logits_per_text = logit_scale * text_features @ all_image_features.T
        #     else:
        #         logits_per_image = logit_scale * all_image_features @ all_text_features.T
        #         logits_per_text = logits_per_image.T
        # else:
        total_loss = None
        for i in range(len(image_features)):
            cur_text_feature = text_features[i] if i<len(text_features) else text_features[-1]
            cur_label = labels[i] if i<len(labels) else labels[-1]
            # print("img.shape",image_features[i].shape,text_features[i].shape,labels[i].shape)
            logits_per_image = logit_scale * image_features[i] @ cur_text_feature.T
            logits_per_text = logit_scale * cur_text_feature @ image_features[i].T
            
            # logits_per_image b b logits_per_text b b
            # calculated ground-truth and cache if enabled
            num_logits = logits_per_image.shape[0] # b
            # labels = torch.eye(num_logits, device=device, dtype=torch.float) # 对角线为1其余为0
            pred_1 = F.log_softmax(logits_per_image,dim=-1) # 对最后一个维度先做softmax后做log
            pred_2 = F.log_softmax(logits_per_text,dim=-1)
            loss_a = F.kl_div(pred_1, cur_label, reduction = 'sum')/num_logits
            loss_b = F.kl_div(pred_2, cur_label, reduction = 'sum')/num_logits
            if total_loss is None:
                total_loss = (loss_a + loss_b)/2
            else:
                total_loss += (loss_a + loss_b)/2
        total_loss/=len(image_features)
        return total_loss

class GlobalLocalLoss(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        
    def forward(self, image_features, report_feature, labels):
        # image_features 4 b d
        # report_features b d
        # labels b b
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        total_loss = None
        for i in range(len(image_features)):
            # print("img.shape",image_features[i].shape,report_feature.shape,labels.shape)
            logits_per_image = logit_scale * image_features[i] @ report_feature.T
            logits_per_text = logit_scale * report_feature @ image_features[i].T
            
            # logits_per_image b b logits_per_text b b
            # calculated ground-truth and cache if enabled
            num_logits = logits_per_image.shape[0] # b
            # labels = torch.eye(num_logits, device=device, dtype=torch.float) # 对角线为1其余为0
            pred_1 = F.log_softmax(logits_per_image,dim=-1) # 对最后一个维度先做softmax后做log
            pred_2 = F.log_softmax(logits_per_text,dim=-1)
            loss_a = F.kl_div(pred_1, labels, reduction = 'sum')/num_logits
            loss_b = F.kl_div(pred_2, labels, reduction = 'sum')/num_logits
            if total_loss is None:
                total_loss = (loss_a + loss_b)/2
            else:
                total_loss += (loss_a + loss_b)/2
        total_loss/=len(image_features)
        return total_loss
