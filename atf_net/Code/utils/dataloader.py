from torch.utils import data
import torch
import os
from PIL import Image
class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        
        self.image_path = []
        self.label_path = []
        self.depths = []
        print(img_root, label_root)
        videos_list = sorted(os.listdir(img_root))
        for video in videos_list:
            if video == '.DS_Store':
                continue
            rgb_path = os.path.join(img_root, video)
            rgb_list = [x for x in sorted(os.listdir(rgb_path)) if x.endswith('.png')]
            # rgb_list = rgb_list[1:]
            for x in rgb_list:
                if x.endswith('.png'):
                    self.image_path.append(os.path.join(rgb_path, x)) 


        videos_list = sorted(os.listdir(label_root))
        for video in videos_list:
            if video == '.DS_Store':
                continue
            rgb_path = os.path.join(label_root, video) + '/gt'
            rgb_list = [x for x in sorted(os.listdir(rgb_path)) if x.endswith('.png')]
            # rgb_list = rgb_list[1:]
            for x in rgb_list:
                if x.endswith('.png'):
                    self.label_path.append(os.path.join(rgb_path, x)) 
        
        # print(self.label_path)
        # print('--'*10)
        # print(self.image_path)


    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')

        gt = Image.open(self.label_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
    