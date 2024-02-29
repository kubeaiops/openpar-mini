
from AttrDataset import get_eval_transform
from config import ArgsNamespace, args_dict
from util import set_seed
from clip import load
from model import TransformerClassifier
from PIL import Image
import torch
import pickle
import time

set_seed(605)
device = "cuda"
args = ArgsNamespace(**args_dict)

def load_model():
    # Select image feature extractor
    clip_model, _ = load("ViT-L/14", device=device, download_root='model')

    # Get dataset information
    dataset_dir = 'dataset/'
    dataset_info = pickle.load(open(dataset_dir + args.dataset + "/pad.pkl", 'rb+'))
    attr_num = len(dataset_info.attributes)
    attributes = dataset_info.attributes

    # Load model
    model = TransformerClassifier(clip_model, attr_num, attributes)
    checkpoint = torch.load(args.trained_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return model, clip_model, attributes

def evaluate_image(model, clip_model, attributes, target_image_file):
    start = time.time()
    print(f'Start - :{start}')

    # Process evaluation image
    eval_transform = get_eval_transform(args)
    img_pil = Image.open(target_image_file)
    if  img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        print (f'image file is not RGB mode, {img_pil.mode} mode. Converted to RGB mode')

    img_transformed = eval_transform(img_pil)

    validator(model=model,
              attributes=attributes,
              image=img_transformed,
              clip_model=clip_model, 
              threshold = args.eval_threshold)

    end = time.time()
    total = end - start
    print(f'The time taken for the test epoch is:{total}')

def validator(model, attributes, image, clip_model, threshold):
    #for attr in attributes:
    #    formatted_attr = f"{attr}"
    #    print("formatted attribute", formatted_attr, end=',')

    valid_probs, valid_logits, final_similarity = valid_image(model, clip_model, image)

    for prob, attribute in zip(valid_probs[0], attributes):
        if prob > threshold:
            print(f"{attribute}: {prob}")

def valid_image(model, clip_model, image):
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.cuda()
        valid_logits, final_similarity = model(image, clip_model=clip_model)
        valid_probs = torch.sigmoid(valid_logits)

    return valid_probs, valid_logits, final_similarity
