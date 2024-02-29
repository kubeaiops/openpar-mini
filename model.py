import torch.nn as nn
import torch
from clip import tokenize
from vit import *
from config import ArgsNamespace, args_dict

args = ArgsNamespace(**args_dict)

class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, attr_num, attributes, dim=768, pretrain_path='model/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
        vit = vit_base()
        vit.load_param(pretrain_path)
        self.norm = vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.dim = dim
        self.text = tokenize(attributes).to("cuda")
        self.bn = nn.BatchNorm1d(self.attr_num)
        fusion_len = self.attr_num + 257 + args.vis_prompt
        if not args.use_mm_former :
            print('Without MM-former, Using MLP Instead')
            self.linear_layer = nn.Linear(fusion_len, self.attr_num)
        else:
            self.blocks = vit.blocks[-args.mm_layers:]
    def forward(self,imgs,clip_model):
        #print("TransformerClassifier - Input image shape:", imgs.shape)
        b_s=imgs.shape[0]
        clip_image_features,all_class,attenmap=clip_model.visual(imgs.type(clip_model.dtype))
        #print("TransformerClassifier - Clip image features shape:", clip_image_features.shape)

        text_features = clip_model.encode_text(self.text).to("cuda").float()
        #print("TransformerClassifier - Text features shape:", text_features.shape)
        if args.use_div:
            final_similarity,logits_per_image = clip_model.forward_aggregate(all_class,text_features)
        else : 
            final_similarity = None
        textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
        x = torch.cat([textual_features,clip_image_features], dim=1)
        
        if args.use_mm_former:
            for blk in self.blocks:
                x = blk(x)
        else :# using linear layer fusion
            x = x.permute(0, 2, 1)
            x= self.linear_layer(x)
            x = x.permute(0, 2, 1)
            
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        
        
        return bn_logits,final_similarity