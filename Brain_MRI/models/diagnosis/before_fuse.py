import torch.nn as nn
import torch

class beforeFuse(nn.Module):
    def __init__(self, config, mode='train'):
        super(beforeFuse, self).__init__()
        self.d_model = config['d_model']
        self.res_linear1=nn.Linear(self.d_model*4, self.d_model)
        self.res_linear2=nn.Linear(self.d_model, self.d_model)
        self.apply(self._init_weights)

    def forward(self,features):
        out_feature=torch.cat(features,dim=2) # b,p,768*4
        out_feature=self.res_linear1(out_feature) 
        out_feature=self.res_linear2(out_feature)# b,p,768
        return out_feature

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