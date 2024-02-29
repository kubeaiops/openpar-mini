args_dict = {
    'dataset': 'PETA',
    'batchsize': 16,
    'epoch': 5,
    'height': 224,
    'width': 224,
    'lr': 0.008,
    'weight_decay': 0.0001,
    'ag_threshold': 0.5,
    'smooth_param': 0.1,
    'use_div': True,
    'use_vismask': True,
    'use_GL': True,
    'use_textprompt': True,
    'use_mm_former': True,
    'mm_layers': 1,
    'div_num': 4,
    'overlap_row': 2,
    'text_prompt': 3,
    'vis_prompt': 50,
    'vis_depth': 24,
    'clip_lr': 0.004,
    'clip_weight_decay': 0.0001,
    'mmformer_update_parameters': ['word_embed', 'weight_layer', 'bn', 'norm'],
    'clip_update_parameters': ['prompt_deep', 'prompt_text_deep', 'part_class_embedding', 'agg_bn', 'softmax_model'],
    'train_split': 'trainval',
    'valid_split': 'test',
    'redirector': True,
    'save_freq': 1,
    'checkpoint': False,
    'trained_model_path': 'model/trained_model.pth',
    'eval_threshold': 0.45
}
class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

