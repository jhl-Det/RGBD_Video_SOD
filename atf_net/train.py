import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import torch.optim as optim
from Code.ours.model import ATFNet
from Code.utils.data_flow_v2 import get_loader,test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
from tqdm import tqdm

#set the device for training                                                                                                                                                                        
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# print('USE GPU ', opt.gpu_id)
os.environ["OMP_NUM_THREADS"] = "1"

DEVICE = torch.device("mps")

# cudnn.benchmark = True

#build the model
model = ATFNet(32, 50)
if "," in opt.gpu_id:
    model = torch.nn.DataParallel(model)
    opt.batchsize *= len(opt.gpu_id.split(","))
    print(f"current batch size={opt.batchsize}")

if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load), strict=False)
    print('load model from ',opt.load)

model.to(DEVICE)
params    = model.parameters()
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

optimizer = torch.optim.AdamW(params, opt.lr)

num_iters = 0
total_iters = 7000

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)

#set the path
train_image_root = opt.rgb_label_root
train_gt_root    = opt.gt_label_root
train_depth_root = opt.depth_label_root

val_image_root   = train_image_root.replace('/train', '/test')
val_gt_root      = opt.val_gt_root
val_depth_root   = opt.val_depth_root
save_path        = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(train_image_root, train_gt_root,train_depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader  = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)
total_step   = len(train_loader)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Config")
logging.info(opt)
logging.info(model)

step = 0
best_mae = 1.0
best_epoch = 0
start_epoch = 1
print(len(train_loader))

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def save_checkpoint(model, optimizer, epoch, save_name):
    total_stat_dict = {}
    total_stat_dict['optimizer'] = optimizer.state_dict()
    total_stat_dict['model'] = model.state_dict()
    total_stat_dict['epoch'] = epoch
    total_stat_dict['scheduler'] = scheduler.state_dict()
    torch.save(total_stat_dict, save_name)

def load_checkpoint(model_name):
    total_stat_dict = torch.load(model_name)
    model.load_state_dict(total_stat_dict['model'])
    optimizer.load_state_dict(total_stat_dict['optimizer'])
    start_epoch = total_stat_dict['epoch'] + 1
    scheduler.load_state_dict(total_stat_dict['scheduler'])
    return start_epoch

def train(train_loader, model, optimizer, epoch, save_path):
    global step, num_iters
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths, flows) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images   = images.to(DEVICE)
            gts      = gts.to(DEVICE)
            depths   = depths.to(DEVICE)
            flows    = flows.to(DEVICE)
            ##
            rgb_out, dep_out, flo_out, ful_out  = model(images, depths, flows)
            
            loss1    = structure_loss(rgb_out, gts) 
            loss2    = structure_loss(dep_out, gts)
            loss3    = structure_loss(flo_out, gts) 
            loss4    = structure_loss(ful_out, gts) 

            loss = loss1 + 0.5 * loss2 + 0.5 * loss3 + loss4
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            scheduler.step()
            step += 1
            num_iters += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 10 == 0 or i == total_step or i==1:
                tmp_str = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], rgb: {:.4f} dep: {:0.4f} motion: {:0.4f} fusion: {:0.4f}'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data, loss4.data)
                print(tmp_str)
                logging.info('#TRAIN#:'+tmp_str)
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        
        if (epoch) % 1 == 0:
            save_checkpoint(model, optimizer, epoch, save_path+'epoch_{}.pth'.format(epoch))
            save_checkpoint(model, optimizer, epoch, save_path+'latest.pth')
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        raise
        
#test function
def val(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in tqdm(range(test_loader.size)):
            image, gt, depth, flow, name, img_for_post = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.cuda()
            flow    = flow.cuda()
            res = model(image, depth, flow)[-1]
            res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res >= 0.5).astype(np.uint8)

            mae_sum += np.mean(np.abs(res-gt))
        
        mae = mae_sum/test_loader.size

        print('Epoch: {} MAE: {}, best MAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch: {} MAE: {}, best MAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
 
if __name__ == '__main__':
    print("Start train...")
    if opt.resume:
        print(f"reume model from {opt.save_path}latest.pth")
        start_epoch = load_checkpoint(opt.save_path+'latest.pth')
    
    for epoch in range(start_epoch, opt.epoch):
        train(train_loader, model, optimizer, epoch,save_path)
        # if epoch > 20:
        #     val(test_loader, model, epoch, save_path)
        logging.info(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]}")
        print(f"Epoch {epoch},LR: {scheduler.get_last_lr()[0]}")
        logging.info("=============================")


