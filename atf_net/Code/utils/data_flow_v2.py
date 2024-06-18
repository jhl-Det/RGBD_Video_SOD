import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

#several data augumentation strategies
def cv_random_flip(img, label,depth,flow):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flow = flow.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth, flow
def randomCrop(image, label, depth, flow):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), flow.crop(random_region)
def randomRotation(image, label, depth, flow):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        flow=flow.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
    return image, label, depth, flow
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
from glob import glob

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        self.trainsize = trainsize
        self.images = []
        self.gts = []
        self.depths = []
        self.flows = []

        self.images = sorted(glob(image_root + "/*/rgb/*png"))
        print(f"found {len(self.images)} images")

        for x in self.images:
            self.gts.append(x.replace("/rgb/", "/gt/"))
            self.depths.append(x.replace("/rgb/", "/depth/"))
            self.flows.append(x.replace("/rgb/", "/").replace("/train/", "/train_flow/"))

        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.binary_loader(self.depths[index])
        flow=self.rgb_loader(self.flows[index])

        
        image,gt,depth, flow =cv_random_flip(image, gt, depth, flow)
        image,gt,depth, flow=randomCrop(image, gt, depth, flow)
        image,gt,depth, flow=randomRotation(image, gt, depth, flow)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        
        image = self.img_transform(image)
        flow = self.img_transform(flow)
        gt = self.gt_transform(gt)
        depth=self.depths_transform(depth)
        
        return image, gt, depth, flow

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images) and len(self.gts)==len(self.flows)
        images = []
        gts = []
        depths=[]
        flows=[]

        for img_path, gt_path, depth_path, flow_depth in zip(self.images, self.gts, self.depths, self.flows):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            flow= Image.open(flow_depth)

            if img.size == gt.size and gt.size==depth.size and gt.size==flow.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                flows.append(flow_depth)

        self.images = images
        self.gts = gts
        self.depths = depths
        self.flows = flows


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=1, pin_memory=False):

    dataset = SalObjDataset(image_root, gt_root, depth_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root,depth_root, testsize):
        self.testsize = testsize
        self.images = []
        self.gts = []
        self.depths = []
        self.flows = []

        self.images = sorted(glob(image_root + "/*/rgb/*png"))
        print(f"found {len(self.images)} images")

        for x in self.images:
            self.gts.append(x.replace("/rgb/", "/gt/"))
            self.depths.append(x.replace("/rgb/", "/depth/"))
            self.flows.append(x.replace("/test/", "/test_flow/").replace("rgb/", ""))

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        flow = self.rgb_loader(self.flows[self.index])
        # print(self.flows[self.index])
        flow = self.transform(flow).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth=self.binary_loader(self.depths[self.index])
        depth=self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')
        name = name[-3] + '/' + name[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, flow, name,np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

