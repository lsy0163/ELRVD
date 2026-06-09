from __future__ import division
import os, time, scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
#from skimage.measure import compare_psnr,compare_ssim
from tensorboardX import SummaryWriter
from models import RViDeNet, RViDeNet_sfb
from utils import *
from video_denoising_models import DnCNN_Fusion, DnCNN_predenoising
from video_denoising_models import FastDVDnet

def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    return img

def generate_alpha(input, lower=1, upper=5):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()
    return ratio

parser = argparse.ArgumentParser(description='Pretrain denoising model')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=2, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=33, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = "/database/iyj0121/result/tcsvt/pretrain_CRVD_rggb_baseline/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

gt_paths1 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-02_raw/*.raw')
gt_paths2 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-09_raw/*.raw')
gt_paths3 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-10_raw/*.raw')
gt_paths4 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-11_raw/*.raw')
gt_paths = gt_paths1 + gt_paths2 + gt_paths3 + gt_paths4

ps = args.patch_size  # patch size for training
batch_size = args.batch_size # batch size for training

log_dir = "/database/iyj0121/tensorboard/pretrain_CRVD_rggb_baseline/"
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

learning_rate = 1e-4

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False

predenoiser = torch.load("/database/iyj0121/dataset/Sony/predenoising_all_ISO/model_epoch700.pth")
for k,v in predenoiser.named_parameters():
    v.requires_grad=False

##################################################################################################################################################################
# device = torch.device("cuda")
# unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
# unet_model = unet_model.cuda()
# noise_generator = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='Unet', device=device, noise_list='shot_read_row1_rowt') #shot_read_uniform_row1_rowt_periodic
# saved_state_dict = torch.load("/database/iyj0121/ckpt/generator/noisemodelUnet_shot_read_row1_rowt_256_color_fourier_final/generatorcheckpoint190_Gloss-0.19685_Dloss-0.57082.pt", map_location='cuda:0')
# new_state_dict = {}
# for k, v in saved_state_dict.items():
#     new_key = k.replace('module.', '') 
#     new_state_dict[new_key] = v
# noise_generator.load_state_dict(new_state_dict) #, strict=False
# for name, param in noise_generator.named_parameters():
#     print(name, param.shape)
##################################################################################################################################################################

denoiser = RViDeNet(predenoiser=predenoiser).cuda()
##################################################################################################################################################################
#denoiser = DnCNN_Fusion(channels=4, num_of_layers=25, nframes=3, center=1).cuda() #수정
#denoiser = DnCNN_predenoising(predenoiser=predenoiser ,channels=4, num_of_layers=20, nframes=3, center=1).cuda() #수정
#denoiser = FastDVDnet().cuda()
##################################################################################################################################################################

initial_epoch = findLastCheckpoint(save_dir=save_dir)  
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1

opt = optim.Adam(denoiser.parameters(), lr = learning_rate)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_raws = [None] * len(gt_paths)

iso_list = [1600,3200,6400,12800, 25600] #1600,3200,6400,12800,25600
a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]
#a_list = [3.579473207859392, 7.045631474798787, 14.320933881640014, 26.678185666447533] #3.513262,6.955588,13.486051,26.585953,52.032536
#g_noise_var_list = [10.536835728053509,19.891667445693656,140.1581801855819,430.5023989990135] #11.917691,38.117816,130.818508,484.539790,1819.818657
# row_noise_var_list = [0.12174298074963856, 0.18673977030485467, 0.764572053691291, 1.8962790397361151] 
#a_list = [1.2857106169195294, 1.819355909636196, 2.5814470195646404, 3.579473207859392, 5.1002582049391325, 7.045631474798787, 10.402380045838742, 14.320933881640014, 20.943255719418207, 26.678185666447533] #3.513262,6.955588,13.486051,26.585953,52.032536
#g_noise_var_list = [1.789327285214302, 0.8366205614907326, 5.842170625067171, 10.536835728053509, 19.89870969466666, 19.891667445693656, 74.56694064745963, 140.1581801855819, 270.1312830659088 ,430.5023989990135] #11.917691,38.117816,130.818508,484.539790,1819.818657
#g_noise_var_list = [1.725756175688214, 0.8181465293547246, 5.581248900710507, 10.021398336585301, 18.87848802315185, 18.870312485208395, 70.34602224026312, 131.66541865038423, 251.46620644355104, 399.6302557873636]
#row_noise_var_list = [0.12174298074963856, 0.18724842541780534,  0.18673977030485467, 0.4777322138274074, 0.764572053691291, 1.2959910701221697, 1.8962790397361151] 

# vdslab dataset
# a_list = [1.807285710250844, 3.552980096370027, 7.354072242415198, 14.865946789586046, 26.027257491695277]
# g_noise_var_list = [3.267389571551398, 10.447538526592822, 35.72686974348538, 160.53305707445782, 437.11474461399774]
#vnt dataset
# a_list = [1.819355909636196, 3.579473207859392, 7.045631474798787, 14.320933881640014, 26.678185666447533]
# g_noise_var_list = [0.8181465293547246, 10.021398336585301, 18.870312485208395, 131.66541865038423 ,399.6302557873636]
#a_list = [26.678185666447533]
#g_noise_var_list = [430.5023989990135]

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(len(gt_paths)/batch_size)
temporal_frames_num = 3
for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    if epoch > 20:
        for g in opt.param_groups:
            g['lr'] = 1e-5
    for batch_id in range(int(len(gt_paths)/batch_size)):
        input_batch_list = []
        gt_batch_list = []
        batch_num = 0
        while batch_num<batch_size:
            ind = np.random.randint(0,len(gt_paths))

            gt_path = gt_paths[ind]

            #select center frame
            ############################## 원본 코드 ##############################
            gt_fn = os.path.basename(gt_path)
            # if gt_fn[2]!='0':
            #     frame_id = int(gt_fn[2:6])
            # elif gt_fn[3]!='0':
            #     frame_id = int(gt_fn[3:6])
            # elif gt_fn[4]!='0':
            #     frame_id = int(gt_fn[4:6])
            # else:
            #     frame_id = int(gt_fn[5])
            ########################################################################
            frame_id = int(gt_fn.split('.')[0])

            if 'MOT17-02_raw' in gt_path:
                if frame_id<2 or frame_id > len(gt_paths1)-2:
                    continue
            if 'MOT17-09_raw' in gt_path:
                if frame_id<2 or frame_id > len(gt_paths2)-2:
                    continue
            if 'MOT17-10_raw' in gt_path:
                if frame_id<2 or frame_id > len(gt_paths3)-2:
                    continue
            if 'MOT17-11_raw' in gt_path:           
                if frame_id<2 or frame_id > len(gt_paths4)-2:
                    continue
            batch_num += 1

            noisy_level = np.random.randint(1,5+1)
            
            a = a_list[noisy_level-1] #0
            g_noise_var = g_noise_var_list[noisy_level-1] #0
            #row_noise_var = row_noise_var_list[noisy_level-1] #0

            input_pack_list = []
            gt_pack_list = []
            H = 1080
            W = 1920
            xx = np.random.randint(0, W - ps*2+1)
            while xx%2!=0:
                xx = np.random.randint(0, W - ps*2+1)
            yy = np.random.randint(0, H - ps*2+1)
            while yy%2!=0:
                yy = np.random.randint(0, H - ps*2+1)

            for shift in range(-1,2):
                gt_frame_name = generate_name(frame_id+shift)
                gt_frame_path = list(gt_path)
                gt_frame_path[-len(gt_fn):] = gt_frame_name
                gt_frame_path = ''.join(gt_frame_path)
                scene_id = gt_paths.index(gt_frame_path)
                if gt_raws[scene_id] is None:
                    #gt_raw = cv2.imread(gt_frame_path,-1)
                    #gt_raws[scene_id] = gt_raw
                    with open(gt_path, 'rb') as f:
                        gt_raw = np.fromfile(f, dtype=np.uint16).reshape((1080, 1920))
                    gt_raws[scene_id] = gt_raw

                gt_raw_full = gt_raws[scene_id]

                gt_patch = gt_raw_full[yy:yy + ps*2, xx:xx + ps*2]

                gt_pack = np.expand_dims(pack_rggb_raw(gt_patch), axis=0)

                #generate noisy raw
                noisy_raw = generate_noisy_raw(gt_patch.astype(np.float32), a, g_noise_var) # original code
                #noisy_raw = generate_noisy_raw_with_row_uniform(gt_patch.astype(np.float32), a, g_noise_var, row_noise_var) # modified code
                #noisy_raw = generate_noisy_raw_with_uniform(gt_patch.astype(np.float32), a, g_noise_var) # modified code
                input_pack = np.expand_dims(pack_rggb_raw(noisy_raw), axis=0)
                input_pack = np.minimum(input_pack, 1.0)
                ####################################################################################
                # with torch.no_grad():
                #     gt_pack_tensor = torch.from_numpy(gt_pack).float().cuda()
                #     gt_pack_tensor = gt_pack_tensor.permute(0,3,1,2)
                #     noisy_raw = noise_generator(gt_pack_tensor)
                # input_pack = torch.from_numpy(noisy_raw.cpu().numpy()).permute(0,3,1,2).cuda()  
                ####################################################################################
                              
                input_pack_list.append(input_pack)
                gt_pack_list.append(gt_pack)

            #input_pack_list_cpu = [pack.cpu().numpy() for pack in input_pack_list]
            input_pack_frames = np.concatenate(input_pack_list, axis=3)
            gt_pack = gt_pack_list[1]
            
            input_batch_list.append(input_pack_frames)
            gt_batch_list.append(gt_pack)
        input_batch = np.concatenate(input_batch_list, axis=0)
        gt_batch = np.concatenate(gt_batch_list, axis=0)
        in_data = torch.from_numpy(input_batch.copy()).permute(0,3,1,2).cuda()
        gt_data = torch.from_numpy(gt_batch.copy()).permute(0,3,1,2).cuda()
        alpha = generate_alpha(in_data)
         
        denoiser.train()
        opt.zero_grad()

        denoised_out = denoiser(in_data.reshape(batch_size,3,4,ps,ps))
        l1_loss = reduce_mean(denoised_out, gt_data)
        
        loss = l1_loss 
        loss.backward()
        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('l1_loss', l1_loss.item(), step)
        print("epoch:%d iter%d loss=%.6f" % (epoch, cnt, loss.data))

    if epoch%1==0:
        torch.save(denoiser, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
