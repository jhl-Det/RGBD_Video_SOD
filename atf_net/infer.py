import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import time
from Code.ours.model import ATFNet
from Code.utils.data_flow_v2 import test_dataset
import datetime as datetime
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='6', help='select gpu id')
parser.add_argument('--test_path',type=str, default='/home/qd/sod/dataset/test',help='test dataset path')
parser.add_argument('--ckpt',type=str, default=None ,help='test dataset path')
parser.add_argument('--save_path',type=str, default="/Users/junhaolin/Downloads/workspace/exps/0_exp", help='test dataset sal map save path')

opt = parser.parse_args()

dataset_path = opt.test_path
 
DEVICE=torch.device("mps")
#load the model
model = ATFNet(32,50)
model.to(DEVICE)
print("loading... from ", opt.ckpt)

model.load_state_dict(torch.load(opt.ckpt)['model'])
model.eval()
print("resnet50 have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
#test
test_datasets = [''] 

save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
mae_sum,sum_fm,sum_sm=0,0,0

image_root  = '/Users/junhaolin/Downloads/research/saliency_detection/dataset_vidsod_100/test/'

gt_root     = dataset_path
depth_root  = dataset_path
test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
print(test_loader)
avg_p, avg_r, img_num, beta2 = 0.0, 0.0, 0.0, 0.3
sum_pre, sum_rec = 0, 0
start_time = time.time()
for i in tqdm(range(test_loader.size)):
    image, gt, depth, flow, name, image_for_post = test_loader.load_data()
    img_dirname, img_name = name.split('/')[0], name.split('/')[1]
    predictions_save_path = os.path.join(save_path, img_dirname) + '/'
    if not os.path.exists(predictions_save_path):
        os.makedirs(predictions_save_path)
    gt      = np.asarray(gt, np.float32)
    gt     /= (gt.max() + 1e-8)
    image   = image.to(DEVICE)
    depth   = depth.to(DEVICE)
    flow    = flow.to(DEVICE)
    pre_res = model(image, depth, flow)
    res     = pre_res[-1]     
    res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res     = res.sigmoid().data.cpu().numpy().squeeze()
    mae_sum += np.mean(np.abs(res - gt))
    res_img = Image.fromarray(res * 255.)
    res_img = res_img.convert("L")
    res_img.save(predictions_save_path+img_name)

end_time = time.time()-start_time
print("FPS=", test_loader.size / end_time)
mae = mae_sum/test_loader.size
print("MAE=", mae)
print("Done!")