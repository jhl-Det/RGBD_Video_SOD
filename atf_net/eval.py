import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
import time
from Code.utils.data_flow_v2 import test_dataset
import datetime as datetime
from Code.ours.baseline import SPNet
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352,help='testing size')
parser.add_argument('--gpu_id', type=str, default='6', help='select gpu id')
parser.add_argument('--test_path', type=str, defalt='/home/qd/sod/dataset/test',help='test dataset path')
parser.add_argument('--ckpt',type=str, default=None, help='test dataset path')
parser.add_argument('--save_path',type=str, default="./cache", help='test dataset sal map save path')

opt = parser.parse_args()
dataset_path = opt.test_path

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ', opt.gpu_id)

#load the model
model = SPNet(32,56)
model.cuda()
print("loading... from",opt.ckpt)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(opt.ckpt))
model.eval()
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
print(f"Save results into{save_path}")

mae_sum = 0

image_root = dataset_path
gt_root = dataset_path
depth_root = dataset_path

test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
print(test_loader)

start_time = time.time()

for i in tqdm(range(test_loader.size)):
    image, gt, depth, flow, name, image_for_post = test_loader.load_data()
    img_dirname, img_name = name.split('/')[0], name.split('/')[1]
    predictions_save_path = os.path.join(save_path, img_dirname) + "/"
    
    if not os.path.exists(predictions_save_path):
        os.makedirs(predictions_save_path)
    gt = np.asarray(gt,np.float32)
    gt /= (gt.max()+ 1e-8)
    image, depth,flow = image.cuda(),depth.cuda(),flow.cuda()
    pre_res = model(image, depth, flow)
    res = pre_res[0]
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res>=0.5).astype(np.uint8)
    mae_sum += np.mean(np.abs(res - gt))
    res_img = Image.fromarray(res * 255.)
    res_img = res_img.convert("L")
    res_img.save(predictions_save_path+ img_name)
    del pre_res
    torch.cuda.empty_cache()

end_time = time.time()- start_time
total_eval_time =str(datetime.timedelta(seconds=int(end_time)))
print("Total evaluation time: fl,Fps:{f}".format(total_eval_time, test_loader.size / end_time))
mae_sum /= test_loader.size
print(mae_sum)
print("Done!")
