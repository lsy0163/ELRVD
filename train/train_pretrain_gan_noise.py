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
from models import RViDeNet
from utils import *

parser = argparse.ArgumentParser(description='Pretrain denoising model')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=33, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = "/database/iyj0121/dataset/SRVD_data/pretrain_noise_GAN_shot_read_uniform_row1_rowt_periodic_debug_SI_SD_nounet_train_nounet/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

gt_paths1 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-02_raw/*.raw')
gt_paths2 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-09_raw/*.raw')
gt_paths3 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-10_raw/*.raw')
gt_paths4 = glob.glob('/database/iyj0121/dataset/SRVD_data/raw_clean/MOT17-11_raw/*.raw')
gt_paths = gt_paths1 + gt_paths2 + gt_paths3 + gt_paths4

ps = args.patch_size  # patch size for training
batch_size = args.batch_size # batch size for training

log_dir = "/database/iyj0121/dataset/SRVD_data_log/"
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

learning_rate = 1e-4

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False

predenoiser = torch.load("/database/iyj0121/dataset/Sony/predenoising_noise_GAN_shot_shot_read_uniform_row1_rowt_periodic_debug_SI_SD_nounet_train_nounet/model_epoch700.pth")
for k,v in predenoiser.named_parameters():
    v.requires_grad=False

##################################################################################################################################################################
#29p4
device = torch.device("cuda")
unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
unet_model = unet_model.cuda()

noise_generator_29p4_SD = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='shot') #shot_read_uniform_row1_rowt_periodic
noise_generator_29p4_SI = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='read_uniform_row1_rowt_periodic') #shot_read_uniform_row1_rowt_periodic

saved_state_dict_SD = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_dependent_noise/29p4_bestgenerator48_KLD0.67815.pt", map_location='cuda:0')
saved_state_dict_SI = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_independent_noise/29p4_bestgenerator66_KLD1.02345.pt", map_location='cuda:0')

new_saved_state_dict_SD = {k.replace('module.', ''): v for k, v in saved_state_dict_SD.items()}
new_saved_state_dict_SI = {k.replace('module.', ''): v for k, v in saved_state_dict_SI.items()}

noise_generator_29p4_SD.load_state_dict(new_saved_state_dict_SD, strict=False)
noise_generator_29p4_SI.load_state_dict(new_saved_state_dict_SI, strict=False)

#24
device = torch.device("cuda")
unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
unet_model = unet_model.cuda()

noise_generator_24_SD = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='shot') #shot_read_uniform_row1_rowt_periodic
noise_generator_24_SI = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='read_uniform_row1_rowt_periodic') #shot_read_uniform_row1_rowt_periodic

saved_state_dict_SD = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_dependent_noise/24_bestgenerator178_KLD0.77161.pt", map_location='cuda:0')
saved_state_dict_SI = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_independent_noise/24_bestgenerator96_KLD1.18518.pt", map_location='cuda:0')

new_saved_state_dict_SD = {k.replace('module.', ''): v for k, v in saved_state_dict_SD.items()}
new_saved_state_dict_SI = {k.replace('module.', ''): v for k, v in saved_state_dict_SI.items()}

noise_generator_24_SD.load_state_dict(new_saved_state_dict_SD, strict=False)
noise_generator_24_SI.load_state_dict(new_saved_state_dict_SI, strict=False)

#18
device = torch.device("cuda")
unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
unet_model = unet_model.cuda()

noise_generator_18_SD = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='shot') #shot_read_uniform_row1_rowt_periodic
noise_generator_18_SI = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='read_uniform_row1_rowt_periodic') #shot_read_uniform_row1_rowt_periodic

saved_state_dict_SD = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_dependent_noise/18_bestgenerator548_KLD3.29731.pt", map_location='cuda:0')
saved_state_dict_SI = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_independent_noise/18_bestgenerator502_KLD0.65887.pt", map_location='cuda:0')

new_saved_state_dict_SD = {k.replace('module.', ''): v for k, v in saved_state_dict_SD.items()}
new_saved_state_dict_SI = {k.replace('module.', ''): v for k, v in saved_state_dict_SI.items()}

noise_generator_18_SD.load_state_dict(new_saved_state_dict_SD, strict=False)
noise_generator_18_SI.load_state_dict(new_saved_state_dict_SI, strict=False)

#12
device = torch.device("cuda")
unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
unet_model = unet_model.cuda()

noise_generator_12_SD = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='shot') #shot_read_uniform_row1_rowt_periodic
noise_generator_12_SI = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='read_uniform_row1_rowt_periodic') #shot_read_uniform_row1_rowt_periodic

saved_state_dict_SD = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_dependent_noise/12_bestgenerator102_KLD2.45242.pt", map_location='cuda:0')
saved_state_dict_SI = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_independent_noise/12_bestgenerator310_KLD0.53847.pt", map_location='cuda:0')

new_saved_state_dict_SD = {k.replace('module.', ''): v for k, v in saved_state_dict_SD.items()}
new_saved_state_dict_SI = {k.replace('module.', ''): v for k, v in saved_state_dict_SI.items()}

noise_generator_12_SD.load_state_dict(new_saved_state_dict_SD, strict=False)
noise_generator_12_SI.load_state_dict(new_saved_state_dict_SI, strict=False)

#6
device = torch.device("cuda")
unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
unet_model = unet_model.cuda()

noise_generator_6_SD = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='shot') #shot_read_uniform_row1_rowt_periodic
noise_generator_6_SI = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='read_uniform_row1_rowt_periodic') #shot_read_uniform_row1_rowt_periodic

saved_state_dict_SD = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_dependent_noise/6_bestgenerator126_KLD5.85304.pt", map_location='cuda:0')
saved_state_dict_SI = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_independent_noise/6_bestgenerator392_KLD0.03781.pt", map_location='cuda:0')

new_saved_state_dict_SD = {k.replace('module.', ''): v for k, v in saved_state_dict_SD.items()}
new_saved_state_dict_SI = {k.replace('module.', ''): v for k, v in saved_state_dict_SI.items()}

noise_generator_6_SD.load_state_dict(new_saved_state_dict_SD, strict=False)
noise_generator_6_SI.load_state_dict(new_saved_state_dict_SI, strict=False)

#27
device = torch.device("cuda")
unet_model = Unet(n_channel_in=4, n_channel_out=4, residual=True, down='conv', up='tconv', activation='selu')
unet_model = unet_model.cuda()

noise_generator_0_SD = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='shot') #shot_read_uniform_row1_rowt_periodic
noise_generator_0_SI = NoiseGenerator2d3d_distributed_ablation(net=unet_model, unet_opts='noUnet', device=device, noise_list='read_uniform_row1_rowt_periodic') #shot_read_uniform_row1_rowt_periodic

saved_state_dict_SD = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_dependent_noise/27_bestgenerator18_KLD0.77930.pt", map_location='cuda:0')
saved_state_dict_SI = torch.load("/database/iyj0121/ckpt/jione/starlight_noUnet/signal_independent_noise/27_bestgenerator62_KLD1.30916.pt", map_location='cuda:0')

new_saved_state_dict_SD = {k.replace('module.', ''): v for k, v in saved_state_dict_SD.items()}
new_saved_state_dict_SI = {k.replace('module.', ''): v for k, v in saved_state_dict_SI.items()}

noise_generator_0_SD.load_state_dict(new_saved_state_dict_SD, strict=False)
noise_generator_0_SI.load_state_dict(new_saved_state_dict_SI, strict=False)

##################################################################################################################################################################

denoiser = RViDeNet(predenoiser=predenoiser).cuda()

initial_epoch = findLastCheckpoint(save_dir=save_dir)  
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1

opt = optim.Adam(denoiser.parameters(), lr = learning_rate)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_raws = [None] * len(gt_paths)

iso_list = [1600,3200,6400,12800,25600] #1600,3200,6400,12800,25600
a_list = [3.513262,6.955588,13.486051,26.585953,52.032536] #3.513262,6.955588,13.486051,26.585953,52.032536
g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657] #11.917691,38.117816,130.818508,484.539790,1819.818657
#noise_list = [noise_generator_29p4, noise_generator_27, noise_generator_24, noise_generator_21, noise_generator_18, noise_generator_15, noise_generator_12]
SI_noise_list = [noise_generator_29p4_SI, noise_generator_24_SI, noise_generator_18_SI, noise_generator_12_SI, noise_generator_6_SI, noise_generator_0_SI]
SD_noise_list = [noise_generator_29p4_SD, noise_generator_24_SD, noise_generator_18_SD, noise_generator_12_SD, noise_generator_6_SD, noise_generator_0_SD]

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

            noisy_level = np.random.randint(1,6+1)
            
            # a = a_list[noisy_level-1] #0
            # g_noise_var = g_noise_var_list[noisy_level-1] #0

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
                # noisy_raw = generate_noisy_raw(gt_patch.astype(np.float32), a, g_noise_var)
                # input_pack = np.expand_dims(pack_rggb_raw(noisy_raw), axis=0)
                # input_pack = np.minimum(input_pack, 1.0)
                ####################################################################################
                #noise_generator = noise_list[noisy_level-1]
                SI_noise_generator = SI_noise_list[noisy_level-1]
                SD_noise_generator = SD_noise_list[noisy_level-1]
                with torch.no_grad():
                    gt_pack_tensor = torch.from_numpy(gt_pack).float().cuda()
                    gt_pack_tensor = gt_pack_tensor.permute(0,3,1,2)
                    #noisy_raw = noise_generator(gt_pack_tensor)
                    noisy_raw = SD_noise_generator(gt_pack_tensor)
                    noisy_raw = SI_noise_generator(noisy_raw)
                input_pack = torch.from_numpy(noisy_raw.cpu().numpy()).permute(0,3,1,2).cuda()  
                ####################################################################################
                              
                input_pack_list.append(input_pack)
                gt_pack_list.append(gt_pack)

            input_pack_list_cpu = [pack.cpu().numpy() for pack in input_pack_list]
            input_pack_frames = np.concatenate(input_pack_list_cpu, axis=3)
            gt_pack = gt_pack_list[1]
            
            input_batch_list.append(input_pack_frames)
            gt_batch_list.append(gt_pack)
        input_batch = np.concatenate(input_batch_list, axis=0)
        gt_batch = np.concatenate(gt_batch_list, axis=0)
        in_data = torch.from_numpy(input_batch.copy()).permute(0,3,1,2).cuda()
        gt_data = torch.from_numpy(gt_batch.copy()).permute(0,3,1,2).cuda()
         
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
