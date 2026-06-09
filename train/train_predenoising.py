from __future__ import division
import os, time, scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import re
import cv2
import argparse
from PIL import Image
#from skimage.measure import compare_psnr,compare_ssim
from models import Predenoiser, NAFNet
from tensorboardX import SummaryWriter
#import isp
from utils import *
import rawpy

parser = argparse.ArgumentParser(description='Training predenoising module')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=700, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = "/database/iyj0121/dataset/4k_vnt/ckpt/predenoising/predenoising_noise_estimation_crvd_method_neif_on_4k_calibration_gain_6_30_hcg__/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

gt_paths = glob.glob('/database/iyj0121/dataset/Sony/long_raw_arw/*.raw')

ps = args.patch_size  # patch size for training

log_dir = "/database/iyj0121/dataset/4k_vnt/ckpt/predenoising_log/"
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

learning_rate = 1e-4

isp = torch.load("/database/iyj0121/dataset/Sony/isp/model_epoch770.pth").cuda()
for k,v in isp.named_parameters():
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

model = Predenoiser().cuda() # original code
#model = NAFNet().cuda()

opt = optim.Adam(model.parameters(), lr = learning_rate)

initial_epoch = findLastCheckpoint(save_dir=save_dir) 
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    model = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1

# Raw data takes long time to load. Keep them in memory after loaded.
gt_raws = [None] * len(gt_paths)

iso_list = [1600,3200,6400,12800] #1600,3200,6400,12800,25600
#a_list = [1.2857106169195294, 1.819355909636196, 2.5814470195646404, 3.579473207859392, 5.1002582049391325, 7.045631474798787, 10.402380045838742, 14.320933881640014, 20.943255719418207, 26.678185666447533] #3.513262,6.955588,13.486051,26.585953,52.032536
# vdslab dataset
# a_list = [1.807285710250844, 3.552980096370027, 7.354072242415198, 14.865946789586046, 26.027257491695277]
# g_noise_var_list = [3.267389571551398, 10.447538526592822, 35.72686974348538, 160.53305707445782, 437.11474461399774]
#vnt dataset
# a_list = [1.819355909636196, 3.579473207859392, 7.045631474798787, 14.320933881640014, 26.678185666447533]
# g_noise_var_list = [0.8181465293547246, 10.021398336585301, 18.870312485208395, 131.66541865038423 ,399.6302557873636]
a_list = [1.7951993073730284,3.7303084136122773,7.30762767432244,15.28845246598329,16.248841017467797]
g_noise_var_list = [6.484396971941335,14.80348894880128,43.66308608884797,143.9721768126246,491.1346945973005]
#g_noise_var_list = [1.789327285214302, 0.8366205614907326, 5.842170625067171, 10.536835728053509, 19.89870969466666, 19.891667445693656, 74.56694064745963, 140.1581801855819, 270.1312830659088 ,430.5023989990135] #11.917691,38.117816,130.818508,484.539790,1819.818657
#dslr method
#g_noise_var_list = [1.725756175688214, 0.8181465293547246, 5.581248900710507, 10.021398336585301, 18.87848802315185, 18.870312485208395, 70.34602224026312, 131.66541865038423, 251.46620644355104, 399.6302557873636]
# a_list = [26.678185666447533]
# g_noise_var_list = [430.5023989990135]
#row_noise_var_list = [0.12174298074963856, 0.18724842541780534,  0.18673977030485467, 0.4777322138274074, 0.764572053691291, 1.2959910701221697, 1.8962790397361151] 

for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    for ind in np.random.permutation(len(gt_paths)):

        gt_path = gt_paths[ind]

        gt_fn = os.path.basename(gt_path)

        scene_id = gt_paths.index(gt_path)

        noisy_level = np.random.randint(1,5+1)
        
        a = a_list[noisy_level-1] #0
        g_noise_var = g_noise_var_list[noisy_level-1] 
        
        if gt_raws[scene_id] is None:
            #gt_raw = cv2.imread(gt_path,-1)
            #gt_raws[scene_id] = gt_raw[1:-1,:]
            with open(gt_path, 'rb') as f:
                gt_raw = np.fromfile(f, dtype=np.uint16).reshape((2848, 4256))  # Adjust shape as needed
            gt_raws[scene_id] = gt_raw

        gt_raw_full = gt_raws[scene_id]

        #Bayer Preserving Augmentation
        aug_mode = np.random.randint(3)
        gt_raw_augmentation = bayer_preserving_augmentation(gt_raw_full, aug_mode)

        H = gt_raw_full.shape[0]
        W = gt_raw_full.shape[1]
       
        if aug_mode == 0:
            W = W - 2
        elif aug_mode == 1:
            H = H - 2
        else:
            exchange = H
            H = W
            W = exchange 

        xx = np.random.randint(0, W - ps*2+1)
        while xx%2!=0:
            xx = np.random.randint(0, W - ps*2+1)
        yy = np.random.randint(0, H - ps*2+1)
        while yy%2!=0:
            yy = np.random.randint(0, H - ps*2+1)
        gt_patch = gt_raw_augmentation[yy:yy + ps*2, xx:xx + ps*2]
        
        gt_pack = np.expand_dims(pack_rggb_raw(gt_patch), axis=0)

        cnt += 1
        #generate noisy raw
        noisy_raw = generate_noisy_raw(gt_patch.astype(np.float32), a, g_noise_var) # original code
        #noisy_raw = generate_noisy_raw_with_row_uniform(gt_patch.astype(np.float32), a, g_noise_var, row_noise_var) # modified code
        #noisy_raw = generate_noisy_raw_with_uniform(gt_patch.astype(np.float32), a, g_noise_var) # modified code
        input_pack = np.expand_dims(pack_rggb_raw(noisy_raw), axis=0)
        input_pack = np.minimum(input_pack, 1.0)
        in_img = torch.from_numpy(input_pack.copy()).permute(0,3,1,2).cuda()
        ##################################################################################
        # with torch.no_grad():
        #     gt_pack_tensor = torch.from_numpy(gt_pack).float().cuda()
        #     gt_pack_tensor = gt_pack_tensor.permute(0,3,1,2)
        #     noisy_raw = noise_generator(gt_pack_tensor)
        ##################################################################################
        
        #in_img = noisy_raw
        gt_img = torch.from_numpy(gt_pack.copy()).permute(0,3,1,2).cuda()

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        writer.add_scalar('loss', loss.item(), epoch*len(gt_paths)+cnt)

        print("epoch:%d iter%d loss=%.3f" % (epoch, cnt, loss.data))

    if epoch%50==0:
        torch.save(model, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
