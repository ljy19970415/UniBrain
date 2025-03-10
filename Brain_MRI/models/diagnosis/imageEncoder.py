
from einops import rearrange
from . import resnet

import torch
from torch import nn
import torch.nn.functional as F


from io import BytesIO

class ModelRes(nn.Module):
    def __init__(self, config):
        super(ModelRes, self).__init__()
        self.resnet = self._get_res_base_model(config['model_type'],config['model_depth'],config['input_W'],
                                                config['input_H'],config['input_D'],config['resnet_shortcut'],
                                                config['no_cuda'],config['gpu_id'],config['pretrain_path'],config['out_feature'])
        # num_ftrs = int(self.resnet.fc.in_features/2)
        # self.res_features = nn.Sequential(*list(self.resnet.children())[:-3])
        # self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.res_l2 = nn.Linear(num_ftrs, 768)

        num_ftrs=int(self.resnet.conv_seg[2].in_features)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-1])
        # # num_ftrs=2048
        out_feature=config['out_feature']
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, out_feature)

    def _get_res_base_model(self,model_type,model_depth,input_W,input_H,input_D,resnet_shortcut,no_cuda,gpu_id,pretrain_path,out_feature):
        if model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 256
        elif model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 512
        elif model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 512
        elif model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048
        elif model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=1)
            fc_input = 2048

        model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=out_feature, bias=True))

        net_dict = model.state_dict()
        model = model.cuda()

        if pretrain_path != 'None':
            print('loading pretrained model {}'.format(pretrain_path))
            pretrain = torch.load(pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict) 
            model.load_state_dict(net_dict) 
            print("-------- pre-train model load successfully --------")
        return model

    def forward(self, images):
        # len(images) 4
        # out_embeds = []
        # out_pools = []
        # for i in range(len(images)):
        img = images.float()
        img = img.cuda()
        batch_size = img.shape[0]
        res_fea = self.res_features(img) #batch_size,feature_size,patch_num,patch_num
        # print(res_fea.shape)
        res_fea = rearrange(res_fea,'b d n1 n2 n3 -> b (n1 n2 n3) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_embed = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_embed,dim=1)
        # out_embeds.append(out_emb)
        # out_pools.append(out_pool)
        return out_embed,out_pool