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
from models import RViDeNet, Restormer
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

parser = argparse.ArgumentParser(description='Finetune denoising model')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=2, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=70, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# save_dir = "/database/iyj0121/ckpt/rvidnet/pretrain_signal_independent_dependent_neif_noise_estimation_gain_only_29p4_final_mot_dataset_predenoising_nafnet__pretrain_sfb_per_recon_layer_no_sc/" #_clean_and_slightly_denoised_gt
save_dir = "/database/iyj0121/ckpt/rvidenet/finetune_signal_independent_dependent_neif_noise_estimation_gain_only_29p4_FastDVDnet_no_predenoising_5frame/"
if not os.path.isdir(save_dir): #
    os.makedirs(save_dir)

ps = args.patch_size  # patch size for training
batch_size = args.batch_size # batch size for training

log_dir = "/database/iyj0121/tensorboard/finetune_signal_independent_dependent_neif_noise_estimation_gain_only_29p4_FastDVDnet_no_predenoising_5frame/"
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False

#predenoiser_restormer = Restormer().cuda()
predenoiser = torch.load("/database/iyj0121/dataset/Sony/predenoising_signal_independent_dependent_neif_noise_estimation_only_gain_29p4/model_epoch700.pth") #"/database/iyj0121/dataset/Sony/predenoising_all_ISO/model_epoch700.pth"
#predenoiser = torch.load("/database/iyj0121/dataset/Sony/predenoising_signal_independent_dslr_dependent_vdslab_calibration_dataset_neif_noise_estimation_gain_6_29p4/model_epoch700.pth")

#checkpoint = state_dict['Restormer']
#predenoiser_restormer.load_state_dict(state_dict)
for k,v in predenoiser.named_parameters():
    v.requires_grad=False

#denoiser = RViDeNet(predenoiser=predenoiser).cuda()
#denoiser = DnCNN_Fusion(channels=4, num_of_layers=25, nframes=3, center=1).cuda()
#denoiser = DnCNN_predenoising(channels=4, num_of_layers=20, predenoiser=predenoiser).cuda()
#denoiser = FastDVDnet().cuda()
denoiser = FastDVDnet().cuda()

initial_epoch = findLastCheckpoint(save_dir=save_dir)  
print('initial epoch: {}'.format(initial_epoch))
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1
else:
    denoiser = torch.load(os.path.join("/database/iyj0121/dataset/SRVD_data/pretrain_signal_independent_dependent_neif_noise_estimation_gain_only_29p4_FastDVDnet_no_predenoising_5frame/model_epoch33.pth")).cuda()
    #denoiser = torch.load(os.path.join("/database/iyj0121/ckpt/jione/rvidenet/pretrain_sfb2/model_epoch33.pth")).cuda()

##########################################################################################
#original code
# recon_params1 = list(map(id, denoiser.recon_trunk.parameters()))
# recon_params2 = list(map(id, denoiser.cbam.parameters()))
# recon_params3 = list(map(id, denoiser.conv_last.parameters()))
# base_params = filter(lambda p: id(p) not in recon_params1+recon_params2+recon_params3, denoiser.parameters())

# opt = optim.Adam([{'params': base_params}, {'params': denoiser.recon_trunk.parameters(), 'lr': 1e-5}, {'params': denoiser.cbam.parameters(), 'lr': 1e-5}, {'params': denoiser.conv_last.parameters(), 'lr': 1e-5}], lr = 1e-6)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
##########################################################################################
#new code
base_params = list(denoiser.temp1.parameters()) + list(denoiser.temp2.parameters())
opt = optim.Adam([
    {'params': denoiser.temp1.parameters(), 'lr': 1e-6},
    {'params': denoiser.temp2.parameters(), 'lr': 1e-5}
])
##########################################################################################
train_data_length = 9*1*60

iso_list = [1600,3200,6400,12800,25600] #1600,3200,6400,12800,25600

##########################################################################################
# L0Loss
# total_iter = args.num_epochs * int(train_data_length / args.batch_size)
# l0_loss_fn = L0Loss(total_iter=total_iter, loss_weight=1.0)
##########################################################################################

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(train_data_length/batch_size)
temporal_frames_num = 5
for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    for batch_id in range(int(train_data_length/batch_size)):
        input_batch_list = []
        gt_raw_batch_list = []
        self_consistency_batch_list = []
        noisy_level_batch_list = []
        batch_num = 0
        while batch_num<batch_size:
            batch_num += 1

            scene_ind = np.random.randint(1,9+1)
            frame_ind = np.random.randint(2,58+1)
            #noisy_level = np.random.randint(1,5+1)
            noisy_level = 2
            noisy_frame_index_for_current = np.random.randint(1,88+1)

            noisy_level_batch_list.append(np.expand_dims((noisy_level-1)*9/4, axis=0))

            input_pack_list = []
            gt_raw_pack_list = []
            self_consistency_pack_list = []
            H = 1080
            W = 1920

            xx = np.random.randint(0, W - ps*2+1)
            while xx%2!=0:
                xx = np.random.randint(0, W - ps*2+1)
            yy = np.random.randint(0, H - ps*2+1)
            while yy%2!=0:
                yy = np.random.randint(0, H - ps*2+1)

            for shift in range(-2,3):
                #gt_raw_path = f'/database/iyj0121/dataset/CRVD_data/scene{scene_ind}/ISO{iso_list[noisy_level - 1]}/frame{frame_ind + shift}_clean.raw'
                #gt_raw_path = f"/database/iyj0121/dataset/CRVD_GT/scene{scene_ind}/ISO{iso_list[noisy_level - 1]}/frame{frame_ind + shift}_clean_and_slightly_denoised.raw"
                gt_raw_path = f'/database/iyj0121/dataset/real_video_indoor/scene{scene_ind}/scene{scene_ind}_frame{frame_ind + shift}/cleaned_bayer_frame.raw'
                #gt_raw_path = f'/database/iyj0121/dataset/real_video_indoor/scene{scene_ind}/scene{scene_ind}_frame{frame_ind + shift}/bm3d_denoised_frame.raw'
                gt_raw = np.fromfile(gt_raw_path, dtype=np.uint16).reshape((1080, 1920))
                #gt_raw = cv2.imread('./data/CRVD_data/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.raw'.format(scene_ind, iso_list[noisy_level-1], frame_ind+shift),-1)
                #gt_raw_full = gt_raws[data_id]
                gt_raw_full = gt_raw
                gt_raw_patch = gt_raw_full[yy:yy + ps*2, xx:xx + ps*2]
                gt_raw_pack = np.expand_dims(pack_rggb_raw(gt_raw_patch), axis=0)


                if shift == 0:
                    for self_consistency_index in range(10):
                        #noisy_raw_path = f'/database/iyj0121/dataset/CRVD_data/scene{scene_ind}/ISO{iso_list[noisy_level - 1]}/frame{frame_ind + shift}_noisy{noisy_frame_index_for_current + self_consistency_index}.raw'
                        noisy_raw_path =  f'/database/iyj0121/dataset/real_video_indoor/scene{scene_ind}/scene{scene_ind}_frame{frame_ind + shift}/long/dump_bayer_frame_{(noisy_frame_index_for_current + self_consistency_index):05d}.raw'
                        noisy_raw = np.fromfile(noisy_raw_path, dtype=np.uint16).reshape((1080, 1920)) 
                        #noisy_raw = cv2.imread('./data/CRVD_data/scene{}/ISO{}/frame{}_noisy{}.raw'.format(scene_ind, iso_list[noisy_level-1], frame_ind+shift, noisy_frame_index_for_current+self_consistency_index),-1)
                        noisy_raw_full = noisy_raw
                        noisy_patch = noisy_raw_full[yy:yy + ps*2, xx:xx + ps*2]
                        input_pack = np.expand_dims(pack_rggb_raw(noisy_patch), axis=0)
                        self_consistency_pack_list.append(input_pack)
                    input_pack_list.append(self_consistency_pack_list[2])
                else:
                    noisy_frame_index_for_other = np.random.randint(0, 100)
                    #noisy_raw_path = f'/database/iyj0121/dataset/CRVD_data/scene{scene_ind}/ISO{iso_list[noisy_level - 1]}/frame{frame_ind + shift}_noisy{noisy_frame_index_for_other}.raw'
                    noisy_raw_path = f'/database/iyj0121/dataset/real_video_indoor/scene{scene_ind}/scene{scene_ind}_frame{frame_ind + shift}/long/dump_bayer_frame_{noisy_frame_index_for_other:05d}.raw'
                    noisy_raw = np.fromfile(noisy_raw_path, dtype=np.uint16).reshape((1080, 1920))
                    #noisy_raw = cv2.imread('./data/CRVD_data/scene{}/ISO{}/frame{}_noisy{}.raw'.format(scene_ind, iso_list[noisy_level-1], frame_ind+shift, noisy_frame_index_for_other),-1)
                    noisy_raw_full = noisy_raw
                    noisy_patch = noisy_raw_full[yy:yy + ps*2, xx:xx + ps*2]
                    input_pack = np.expand_dims(pack_rggb_raw(noisy_patch), axis=0)
                    input_pack_list.append(input_pack)
        
                gt_raw_pack_list.append(gt_raw_pack)
     
            self_consistency_pack_frames = np.concatenate(self_consistency_pack_list, axis=3)
            input_pack_frames = np.concatenate(input_pack_list, axis=3)
            gt_raw_pack = gt_raw_pack_list[2]

            input_batch_list.append(input_pack_frames)
            gt_raw_batch_list.append(gt_raw_pack)
            self_consistency_batch_list.append(self_consistency_pack_frames)

        input_batch = np.concatenate(input_batch_list, axis=0)
        gt_raw_batch = np.concatenate(gt_raw_batch_list, axis=0)
        self_consistency_batch = np.concatenate(self_consistency_batch_list, axis=0)
        noisy_level_batch = np.expand_dims(np.expand_dims(np.expand_dims(np.concatenate(noisy_level_batch_list, axis=0), axis=1), axis=2), axis=3)

        in_data = torch.from_numpy(input_batch.copy()).permute(0,3,1,2).cuda()
        gt_raw_data = torch.from_numpy(gt_raw_batch.copy()).permute(0,3,1,2).cuda()
        self_consistency_data = torch.from_numpy(self_consistency_batch.copy()).permute(0,3,1,2).cuda()
        noisy_level_data = torch.from_numpy(noisy_level_batch.copy()).float().cuda()
         
        denoiser.train()
        opt.zero_grad()

        denoised_out = denoiser(in_data.reshape(batch_size,5,4,ps,ps))
        #print(self_consistency_data.shape)

        denoised_out1 = denoiser(self_consistency_data.reshape(batch_size,10,4,ps,ps)[:,0:5,:,:,:])
        denoised_out2 = denoiser(self_consistency_data.reshape(batch_size,10,4,ps,ps)[:,5:10,:,:,:])
        #alpha = generate_alpha(gt_raw_data)
        #print(alpha)

        #raw l1 loss
        raw_l1_loss = reduce_mean(denoised_out, gt_raw_data) + reduce_mean(denoised_out1, gt_raw_data)*0.1 + reduce_mean(denoised_out2, gt_raw_data)*0.1 #original code
        #raw_l1_loss = l0_loss_fn(denoised_out, gt_raw_data, curr_iter=step) + l0_loss_fn(denoised_out1, gt_raw_data, curr_iter=step)*0.1 + l0_loss_fn(denoised_out2, gt_raw_data, curr_iter=step)*0.1
        #srgb l1 loss
        denoised_output_isped = isp(denoised_out)
        gt_isped = isp(gt_raw_data) 
        srgb_l1_loss = reduce_mean(denoised_output_isped, gt_isped) #original code
        #srgb_l1_loss = l0_loss_fn(denoised_output_isped, gt_isped, curr_iter=step)
        # self consistency loss to solve flickering
        self_consistency_loss = reduce_mean_with_weight(denoised_out1, denoised_out2, noisy_level_data)
        
        #loss = (1-alpha) * ( raw_l1_loss + self_consistency_loss + 0.5*srgb_l1_loss)
        loss = raw_l1_loss + self_consistency_loss + 0.5*srgb_l1_loss
        loss = loss.mean()
        loss.backward()
        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('raw_l1_loss', raw_l1_loss.item(), step)
        writer.add_scalar('srgb_l1_loss', srgb_l1_loss.item(), step)
        writer.add_scalar('self_consistency_loss', self_consistency_loss.item(), step)
        #writer.add_scalar('alpha', alpha.mean().item(), step)
        print("epoch:%d iter%d loss=%.6f" % (epoch, cnt, loss.data))

        #scheduler.step(loss)
 
    if epoch%1==0:
        torch.save(denoiser, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
