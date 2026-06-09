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
from tensorboardX import SummaryWriter
from models import RViDeNet, Restormer, Predenoiser
from utils import *
#from dncnn3d import DnCNN_Fusion, DnCNN_predenoising
#from fastdvdnet import FastDVDnet

parser = argparse.ArgumentParser(description='Finetune denoising model')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=2, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=3000, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = "/database/iyj0121/ckpt/lightweight/unet_pretrain_mot_pack_debug/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

ps = args.patch_size  
batch_size = args.batch_size 

log_dir = "/database/iyj0121/tensorboard/restormer/"
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False

# predenoiser = torch.load("/database/iyj0121/dataset/Sony/predenoising_signal_independent_dependent_neif_noise_estimation_only_gain_29p4/model_epoch700.pth").cuda()
# for k,v in predenoiser.named_parameters():
#     v.requires_grad=False

denoiser = Predenoiser().cuda()

initial_epoch = findLastCheckpoint(save_dir=save_dir)  
print('initial epoch: {}'.format(initial_epoch))
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1
else:
    denoiser = torch.load(os.path.join("/database/iyj0121/dataset/4k_vnt/ckpt/pretrain/pretrain_noise_estimation_crvd_method_neif_on_4k_calibration_gain_6_30_hcg_unet/model_epoch33.pth")).cuda() 
    print('load pretrain model')


initial_epoch = findLastCheckpoint(save_dir=save_dir)  
print('initial epoch: {}'.format(initial_epoch))
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1
else:
    pass
    #denoiser = torch.load(os.path.join("/database/iyj0121/dataset/SRVD_data/pretrain/model_epoch33.pth")).cuda()

opt = optim.Adam(denoiser.parameters(), lr=1e-6)

train_data_length = 6*5*7
#iso_list = [1600,3200,6400,12800,25600]
gain_list = [6, 12, 18, 24, 30] 

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(train_data_length/batch_size)
temporal_frames_num = 1

for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    for batch_id in range(int(train_data_length/batch_size)):
        input_batch_list = []
        gt_raw_batch_list = []
        batch_num = 0
        while batch_num < batch_size:
            batch_num += 1

            scene_ind = np.random.randint(2,6+1)
            frame_ind = np.random.randint(2,6+1)
            noisy_level = np.random.randint(1,5+1)
            noisy_frame_index_for_current = np.random.randint(1,230+1)

            #noisy_level_batch_list.append(np.expand_dims((noisy_level-1)*9/4, axis=0))

            input_pack_list = []
            gt_raw_pack_list = []
            H = 2160
            W = 3840

            xx = np.random.randint(0, W - ps*2+1)
            while xx % 2 != 0:
                xx = np.random.randint(0, W - ps*2+1)
            yy = np.random.randint(0, H - ps*2+1)
            while yy % 2 != 0:
                yy = np.random.randint(0, H - ps*2+1)

            #gt_raw_path = f'/database/iyj0121/dataset/real_video_indoor/scene{scene_ind}/scene{scene_ind}_frame{frame_ind}/cleaned_bayer_frame.raw'
            gt_raw_path = f'/database/iyj0121/dataset/4k_vnt/VNT_241120_pair_video/scene{scene_ind}/scene{scene_ind}_frame{frame_ind}_{gain_list[noisy_level-1]}db/cleaned_bayer_frame.raw'
            gt_raw = np.fromfile(gt_raw_path, dtype=np.uint16).reshape((2160, 3840))
            gt_raw_full = gt_raw
            gt_raw_patch = gt_raw_full[yy:yy + ps*2, xx:xx + ps*2]
            gt_raw_pack = np.expand_dims(pack_rggb_raw_4k(gt_raw_patch), axis=0)

            noisy_raw_path = f'/database/iyj0121/dataset/4k_vnt/VNT_241120_pair_video/scene{scene_ind}/scene{scene_ind}_frame{frame_ind}_{gain_list[noisy_level-1]}db/dump_bayer_frame_{noisy_frame_index_for_current:05d}.raw'
            noisy_raw = np.fromfile(noisy_raw_path, dtype=np.uint16).reshape((2160, 3840))
            noisy_patch = noisy_raw[yy:yy + ps*2, xx:xx + ps*2]
            input_pack = np.expand_dims(pack_rggb_raw_4k(noisy_patch), axis=0)

            input_pack_list.append(input_pack)
            gt_raw_pack_list.append(gt_raw_pack)

            input_pack_frames = np.concatenate(input_pack_list, axis=3)
            gt_raw_pack = gt_raw_pack_list[0]

            input_batch_list.append(input_pack_frames)
            gt_raw_batch_list.append(gt_raw_pack)

        input_batch = np.concatenate(input_batch_list, axis=0)
        gt_raw_batch = np.concatenate(gt_raw_batch_list, axis=0)

        in_data = torch.from_numpy(input_batch.copy()).permute(0,3,1,2).cuda()
        gt_raw_data = torch.from_numpy(gt_raw_batch.copy()).permute(0,3,1,2).cuda()

        denoiser.train()
        opt.zero_grad()

        denoised_out = denoiser(in_data.reshape(batch_size,4,ps,ps))

        # raw l1 loss
        raw_l1_loss = reduce_mean(denoised_out, gt_raw_data)
        # srgb l1 loss
        denoised_output_isped = isp(denoised_out)
        gt_isped = isp(gt_raw_data) 
        srgb_l1_loss = reduce_mean(denoised_output_isped, gt_isped)

        loss = raw_l1_loss + 0.5 * srgb_l1_loss
        loss = loss.mean()
        loss.backward()
        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('raw_l1_loss', raw_l1_loss.item(), step)
        #writer.add_scalar('srgb_l1_loss', srgb_l1_loss.item(), step)
        print("epoch:%d iter%d loss=%.6f" % (epoch, cnt, loss.data))

    if epoch % 1 == 0:
        torch.save(denoiser, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
