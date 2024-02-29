import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time

from batch_engine import valid_trainer, batch_trainer
from config import ArgsNamespace, args_dict
from AttrDataset import MultiModalAttrDataset, get_transform

from loss import *
from model import *
from util import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus, get_pedestrian_metrics,count_parameters

from scheduler import make_optimizer, create_scheduler,make_scheduler

from clip import load
from clip_model import *


set_seed(605)
device = "cuda"
clip_model, ViT_preprocess = load("ViT-L/14", device=device, download_root='/home/jerryum/works/OpenPAR/')
args = ArgsNamespace(**args_dict)                                       

def main(args):

    if args.checkpoint==False :
        print(time_str())
        pprint.pprint(OrderedDict(args.__dict__))
        print('-' * 60)
        print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args) 
    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm) 
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm) 
    labels = train_set.label
    sample_weight = labels.mean(0)
    print ('sample weight', sample_weight)


    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')
    

    print("start loading model: ", args.trained_model_path)

    checkpoint = torch.load(args.trained_model_path)
    print('checkpoint keys:', checkpoint.keys())

    #ViT_model=build_model(checkpoint['ViT_model'])

    model = TransformerClassifier(clip_model,train_set.attr_num,train_set.attributes)


    #CUDA_VISIBLE_DEVICES=0 python eval.py RAPV1 --checkpoint --dir ./logs/RAPV1/2023-10-17_19_36_32/epoch23.pth --use_div --use_vismask --vis_prompt 50 --use_GL --use_textprompt --use_mm_former 
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
        #ViT_model=ViT_model.cuda()
    
    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    validator(model=model,
            valid_loader=valid_loader,
            clip_model=clip_model,
            criterion=criterion,
            args=args)


attributes = [
    'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
    'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
    'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
    'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
    'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',
    'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
    'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
    'A male pedestrian'
]


def validator(model, valid_loader, clip_model, criterion, args):
    for attr in attributes:
        formatted_attr = f"{attr}"
        print("formatted attribute", formatted_attr, end=',') 
        
    start=time.time()
    valid_loss, valid_gt, valid_probs, image_names = valid_trainer(
        model=model,
        clip_model=clip_model,
        valid_loader=valid_loader,
        criterion=criterion,
        args=args
    )

    valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
    
    print('valid_loss: {:.4f}'.format(valid_loss))
    print('ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
            'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                valid_result.instance_f1))     
    ma = []
    acc = []
    f1 = []

    valid_probs = valid_probs > args.eval_threshold
    print (f'valide_probs after > {args.eval_threshold}', valid_probs.size, valid_probs)

    pred_attrs=[[] for _ in range(len(image_names))]
    gt_attrs=[[] for _ in range(len(image_names))]
    
    for pidx in range(len(image_names)):
        for aidx in range(len(attributes)):
            if valid_probs[pidx][aidx] : 
                pred_attrs[pidx].append(attributes[aidx])
            if valid_gt[pidx][aidx] : 
                gt_attrs[pidx].append(attributes[aidx])
    
    # Open a text file for writing mode
    with open('preds_img_attrs.txt', 'w') as file:
        # Traverse the key-value pairs of the dictionary and write them to a text file line by line
        for pidx in range(len(image_names)):
            file.write(f'{image_names[pidx]}: {pred_attrs[pidx]}\n')   
    
    #Open a text file for writing mode
    with open('gt_img_attrs.txt', 'w') as file:
        # Traverse the key-value pairs of the dictionary and write them to a text file line by line
        for pidx in range(len(image_names)):
            file.write(f'{image_names[pidx]}: {gt_attrs[pidx]}\n')              
    end=time.time()
    total=end-start 
    print(f'The time taken for the test epoch is:{total}')                 

if __name__ == '__main__':

    main(args)