import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
np.random.seed(0)
random.seed(0)

# note: ref by annotation.md
attr_words = [
    'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
    'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
    'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
    'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
    'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',
    'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
    'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
    'A male pedestrian'
] # 35 words

masked=[[1 for _ in range(35)],
        [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]]

neg_attr_words = [
   'head without hat','head without muffler','head is not nothing','head without sunglasses','head is not long hair',
   'upper is not casual', 'upper is not formal', 'upper is not jacket', 'upper is not logo', 'upper is not plaid', 
   'upper is not short sleeve', 'upper is not thin stripes', 'upper is not t-shirt','upper is not other','upper is not v-neck',
   'lower is not Casual', 'lower is not Formal', 'lower is not Jeans', 'lowe is notr Shorts', 'lower is not Short Skirt','lower is not Trousers',
   'shoes is not Leather', 'shoes is not Sandals', 'shoes is not other', 'shoes is not sneaker',
   'attach is not Backpack', 'attach is not Other', 'attach is not messenger bag', 'attach is not nothing', 'attach is not plastic bags',
   'age over 30','age is not between in 30 to 45','age is not between in 45 to 60','age less 60',
   'female'
] # 35 words

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16] # 35 # group_order list is defined. It specifies the order in which the attributes will be arranged.


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))

    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    
    # Generate image names for 19000 images
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]

    # Extract raw attribute names from the PETA data i.g. ['personalLess30', 'personalLess45',...]
    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]

    # (19000, 105)
    # Extract raw labels for attributes from the PETA data. The 1-4th columns are group information
    raw_label = peta_data['peta'][0][0][0][:, 4:]
    
    # Reorder the labels based on the specified group order
    dataset.label = raw_label[:, group_order]

    # (19000, 35)
    dataset.attr_name = [raw_attr_name[i] for i in group_order]

    # Define attribute words and negative attribute words (not defined in the provided snippet)
    dataset.attributes=attr_words   
    dataset.neg_attr_words=neg_attr_words
    dataset.expand_pos_attr_words=None
    dataset.expand_neg_attr_words=None
    breakpoint()

    # Set label index information for evaluation
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = group_order

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []
    
    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)

        """
        dataset.pkl File containing only key attributes 35 label
        dataset_all.pkl file containing all attributes 105 label
        """
    with open(os.path.join(save_dir, 'pad.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = 'dataset/PETA/'
    generate_data_description(save_dir)
