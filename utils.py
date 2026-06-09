from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
from scipy.stats import poisson
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import time

def pack_gbrg_raw(raw):
    #pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def depack_gbrg_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2*i,2*j]=raw[0,i,j,3]
            output[2*i,2*j+1]=raw[0,i,j,2]
            output[2*i+1,2*j]=raw[0,i,j,0]
            output[2*i+1,2*j+1]=raw[0,i,j,1]
    return output

def depack_rggb_raw(packed_raw):
    H, W, _ = packed_raw.shape
    output = np.zeros((H * 2, W * 2), dtype=packed_raw.dtype)

    output[0:H*2:2, 0:W*2:2] = packed_raw[:, :, 0]  # R
    output[0:H*2:2, 1:W*2:2] = packed_raw[:, :, 1]  # G1
    output[1:H*2:2, 1:W*2:2] = packed_raw[:, :, 2]  # G2
    output[1:H*2:2, 0:W*2:2] = packed_raw[:, :, 3]  # B

    return output

def depack_rggb_raw_crvd(packed_raw):
    H, W, _ = packed_raw.shape
    output = np.zeros((H * 2, W * 2), dtype=packed_raw.dtype)

    output[0:H*2:2, 0:W*2:2] = packed_raw[:, :, 0]  # R
    output[0:H*2:2, 1:W*2:2] = packed_raw[:, :, 1]  # G1
    output[1:H*2:2, 1:W*2:2] = packed_raw[:, :, 2]  # G2
    output[1:H*2:2, 0:W*2:2] = packed_raw[:, :, 3]  # B

    return output

def unpack_raw(raw):
    # Unpack 4-channel image to Bayer pattern
    H, W, _ = raw.shape
    im = np.zeros((H*2, W*2), dtype=np.float32)

    # Reconstruct Bayer pattern from 4 channels
    im[0:H*2:2, 0:W*2:2] = raw[:, :, 0]
    im[0:H*2:2, 1:W*2:2] = raw[:, :, 1]
    im[1:H*2:2, 1:W*2:2] = raw[:, :, 2]
    im[1:H*2:2, 0:W*2:2] = raw[:, :, 3]

    return im.astype(np.uint16)

def pack_rggb_raw(raw):
    #pack RGGB Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def pack_rggb_raw_4k(raw):
    #pack RGGB Bayer raw to 4 channels
    black_level = 200
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def pack_rggb_raw_vnt(raw):
    #pack RGGB Bayer raw to 4 channels
    black_level = 0
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)
    #im = im / 4095.0

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def generate_noisy_raw(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_noisy_raw_vnt(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_noisy_raw_with_row(gt_raw, a, b, row_noise_var):
    """
    a: sigma_s^2
    b: sigma_r^2
    row_noise_var: row noise variance
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    row_noise = np.random.normal(0, (row_noise_var), size=(gt_raw.shape[0], 1))
    row_noise = np.repeat(row_noise, gt_raw.shape[1], axis=1) 
    noisy_img = poisson_noisy_img + gaussian_noise + row_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_noisy_raw_with_row_uniform(gt_raw, a, b, row_noise_var, q=2.9/4096):
    """
    a: sigma_s^2
    b: sigma_r^2
    row_noise_var: row noise variance
    q: uniform noise amplitude based on ADC step (default for 12-bit with 2.9V range)
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    row_noise = np.random.normal(0, (row_noise_var), size=(gt_raw.shape[0], 1))
    row_noise = np.repeat(row_noise, gt_raw.shape[1], axis=1) 
    uniform_noise = np.random.uniform(-q/2, q/2, size=gt_raw.shape)
    noisy_img = poisson_noisy_img + gaussian_noise + row_noise + uniform_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_noisy_raw_with_uniform(gt_raw, a, b, q=2.9/4096):
    """
    a: sigma_s^2
    b: sigma_r^2
    row_noise_var: row noise variance
    q: uniform noise amplitude based on ADC step (default for 12-bit with 2.9V range)
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    #row_noise = np.random.normal(0, (row_noise_var), size=(gt_raw.shape[0], 1))
    #row_noise = np.repeat(row_noise, gt_raw.shape[1], axis=1) 
    uniform_noise = np.random.uniform(-q/2, q/2, size=gt_raw.shape)
    noisy_img = poisson_noisy_img + gaussian_noise + uniform_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_noisy_raw_with_row_uniform_vnt(gt_raw, a, b, row_noise_var, q=2.9/4096):
    """
    a: sigma_s^2
    b: sigma_r^2
    row_noise_var: row noise variance
    q: uniform noise amplitude based on ADC step (default for 12-bit with 2.9V range)
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    row_noise = np.random.normal(0, (row_noise_var), size=(gt_raw.shape[0], 1))
    row_noise = np.repeat(row_noise, gt_raw.shape[1], axis=1) 
    uniform_noise = np.random.uniform(-q/2, q/2, size=gt_raw.shape)
    noisy_img = poisson_noisy_img + gaussian_noise + row_noise + uniform_noise
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_noisy_raw_with_uniform_vnt(gt_raw, a, b, q=2.9/4096):
    """
    a: sigma_s^2
    b: sigma_r^2
    row_noise_var: row noise variance
    q: uniform noise amplitude based on ADC step (default for 12-bit with 2.9V range)
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    #row_noise = np.random.normal(0, (row_noise_var), size=(gt_raw.shape[0], 1))
    #row_noise = np.repeat(row_noise, gt_raw.shape[1], axis=1) 
    uniform_noise = np.random.uniform(-q/2, q/2, size=gt_raw.shape)
    noisy_img = poisson_noisy_img + gaussian_noise + uniform_noise
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

# def generate_name(number):
#     name = list('000000_raw.tiff')
#     num_str = str(number)
#     for i in range(len(num_str)):
#         name[5-i] = num_str[-(i+1)]
#     name = ''.join(name)
#     return name
def generate_name(number):
    return f"{number:06d}.raw" #dump_bayer_frame_

def generate_name_vnt(number):
    return f"dump_bayer_frame_{number:05d}.raw"

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

def reduce_mean_with_weight(im1, im2, noisy_level_data):
    result = torch.abs(im1 - im2) * noisy_level_data * 0.1
    return result.mean()

def preprocess(raw):
    input_full = raw.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full)
    input_full = input_full.cuda()
    return input_full

def postprocess(output):
    output = output.cpu()
    output = output.detach().numpy().astype(np.float32)
    output = np.transpose(output, (0, 2, 3, 1))
    output = np.clip(output,0,1)
    return output

def postprocess_bsvd(output):
    # 출력 차원을 확인
    
    # 5D 텐서인 경우 차원을 조정
    if output.ndim == 5:  # (1, 3, 4, 256, 256)
        output = output.permute(0, 1, 2, 3, 4)  # (1, 256, 256, 4, 3)
        output = output.squeeze(0)  # (256, 256, 4, 3)
        output = output.cpu().numpy()  # 텐서를 numpy 배열로 변환
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    return output

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def bayer_preserving_augmentation(raw, aug_mode):
    if aug_mode == 0:  # horizontal flip
        aug_raw = np.flip(raw, axis=1)[:,1:-1]
    elif aug_mode == 1: # vertical flip
        aug_raw = np.flip(raw, axis=0)[1:-1,:]
    else:  # random transpose
        aug_raw = np.transpose(raw, (1, 0))
    return aug_raw

def test_big_size_raw(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    
    test_result = np.zeros((input_data.shape[0],H,W,4))
    t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,h_begin:h_end,w_begin:w_end,:]               
            test_patch = preprocess(test_patch)               
            with torch.no_grad():
                output_patch = denoiser(test_patch.reshape(1,3,4,patch_h,patch_w))
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        test_patch = input_data[:,h_begin:h_end,-patch_w:,:]         
        test_patch = preprocess(test_patch)
        with torch.no_grad():
            output_patch = denoiser(test_patch.reshape(1,3,4,patch_h,patch_w))
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,-patch_h:,w_begin:w_end,:]               
        test_patch = preprocess(test_patch)               
        with torch.no_grad():
            output_patch = denoiser(test_patch.reshape(1,3,4,patch_h,patch_w))
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    test_patch = input_data[:,-patch_h:,-patch_w:,:]         
    test_patch = preprocess(test_patch)
    with torch.no_grad():
        output_patch = denoiser(test_patch.reshape(1,3,4,patch_h,patch_w))
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
   
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result


def test_big_size_raw_5frame(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    
    test_result = np.zeros((input_data.shape[0],H,W,4))
    t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,h_begin:h_end,w_begin:w_end,:]               
            test_patch = preprocess(test_patch)               
            with torch.no_grad():
                output_patch = denoiser(test_patch.reshape(1,5,4,patch_h,patch_w))
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        test_patch = input_data[:,h_begin:h_end,-patch_w:,:]         
        test_patch = preprocess(test_patch)
        with torch.no_grad():
            output_patch = denoiser(test_patch.reshape(1,5,4,patch_h,patch_w))
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,-patch_h:,w_begin:w_end,:]               
        test_patch = preprocess(test_patch)               
        with torch.no_grad():
            output_patch = denoiser(test_patch.reshape(1,5,4,patch_h,patch_w))
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    test_patch = input_data[:,-patch_h:,-patch_w:,:]         
    test_patch = preprocess(test_patch)
    with torch.no_grad():
        output_patch = denoiser(test_patch.reshape(1,5,4,patch_h,patch_w))
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
   
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def test_big_size_raw_1frame(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    
    test_result = np.zeros((input_data.shape[0],H,W,4))
    t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,h_begin:h_end,w_begin:w_end,:]               
            test_patch = preprocess(test_patch)               
            with torch.no_grad():
                output_patch = denoiser(test_patch.reshape(1,4,patch_h,patch_w))
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        test_patch = input_data[:,h_begin:h_end,-patch_w:,:]         
        test_patch = preprocess(test_patch)
        with torch.no_grad():
            output_patch = denoiser(test_patch.reshape(1,4,patch_h,patch_w))
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,-patch_h:,w_begin:w_end,:]               
        test_patch = preprocess(test_patch)               
        with torch.no_grad():
            output_patch = denoiser(test_patch.reshape(1,4,patch_h,patch_w))
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    test_patch = input_data[:,-patch_h:,-patch_w:,:]         
    test_patch = preprocess(test_patch)
    with torch.no_grad():
        output_patch = denoiser(test_patch.reshape(1,4,patch_h,patch_w))
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
   
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def test_big_size_raw_emvd(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    
    test_result = np.zeros((input_data.shape[0],H,W,4))
    t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,h_begin:h_end,w_begin:w_end,:]               
            test_patch = preprocess(test_patch)   
            #print(test_patch.shape)            
            with torch.no_grad():
                #test_patch = test_patch.squeeze(0).reshape(25,4,patch_h,patch_w)
                #print(test_patch.shape)
                _, _, _, _, output_patch = denoiser(test_patch)
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   
    
        test_patch = input_data[:,h_begin:h_end,-patch_w:,:]         
        test_patch = preprocess(test_patch)
        with torch.no_grad():
            #test_patch = test_patch.squeeze(0).reshape(25,4,patch_h,patch_w)
           _, _, _, _, output_patch = denoiser(test_patch)
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,4))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,-patch_h:,w_begin:w_end,:]               
        test_patch = preprocess(test_patch)               
        with torch.no_grad():
            #test_patch = test_patch.squeeze(0).reshape(25,4,patch_h,patch_w)
            _, _, _, _, output_patch = denoiser(test_patch)
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    test_patch = input_data[:,-patch_h:,-patch_w:,:]         
    test_patch = preprocess(test_patch)
    with torch.no_grad():
        #test_patch = test_patch.squeeze(0).reshape(25,4,patch_h,patch_w)
        _, _, _, _, output_patch = denoiser(test_patch)
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
   
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def pack_gbrg_raw_for_compute_ssim(raw):

    im = raw.astype(np.float32)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def compute_ssim_for_packed_raw(raw1, raw2):
    raw1_pack = pack_gbrg_raw_for_compute_ssim(raw1)
    raw2_pack = pack_gbrg_raw_for_compute_ssim(raw2)
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)

    return test_raw_ssim/4


def test_big_size_raw_bsvd(input_data, denoiser, patch_h=256, patch_w=256, patch_h_overlap=64, patch_w_overlap=64):
    
    print('Input data shape:', input_data.shape)
    H = input_data.shape[1]
    W = input_data.shape[2]

    test_result = np.zeros((input_data.shape[0], 3, 4, H, W))
    t0 = time.time()
    
    h_index = 1
    while (patch_h * h_index - patch_h_overlap * (h_index - 1)) < H:
        h_begin = patch_h * (h_index - 1) - patch_h_overlap * (h_index - 1)
        h_end = patch_h * h_index - patch_h_overlap * (h_index - 1)
        
        w_index = 1
        while (patch_w * w_index - patch_w_overlap * (w_index - 1)) < W:
            w_begin = patch_w * (w_index - 1) - patch_w_overlap * (w_index - 1)
            w_end = patch_w * w_index - patch_w_overlap * (w_index - 1)

            test_patch = input_data[:, h_begin:h_end, w_begin:w_end, :]
            #print('Test patch shape:', test_patch.shape)
            test_patch = preprocess(test_patch)  
            test_patch = test_patch.reshape(1, 3, 4, patch_h, patch_w)

            with torch.no_grad():
                output_patch = denoiser(test_patch)

            print('Output patch shape:', output_patch.shape)
            output_patch = output_patch.cpu().numpy()
            #test_patch_result = postprocess_bsvd(output_patch)

            test_result[:, :, :, h_begin:h_end, w_begin:w_end] = output_patch
            
            w_index += 1
        
        h_index += 1
    
    t1 = time.time()
    print('Total running time: {:.2f} s'.format(t1 - t0))
    
    return test_result

######################################################################################
# The following code is from workspace_j/starlight/models/unet.py
######################################################################################

def split_into_patches2d(x, patch_size = 64):
    patches = torch.empty([1,x.shape[1],patch_size,patch_size], device = x.device)
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', residual=True, activation='leakyrelu', transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        if self.norm1:
            x = self.norm1(x)

        x = self.actfun1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]

        x = self.actfun2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x

class Unet(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu'):
        super(Unet, self).__init__()

        self.residual = residual

        if down == 'maxpool':
            self.down1 = nn.MaxPool2d(kernel_size=2)
            self.down2 = nn.MaxPool2d(kernel_size=2)
            self.down3 = nn.MaxPool2d(kernel_size=2)
            self.down4 = nn.MaxPool2d(kernel_size=2)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool2d(kernel_size=2)
            self.down2 = nn.AvgPool2d(kernel_size=2)
            self.down3 = nn.AvgPool2d(kernel_size=2)
            self.down4 = nn.AvgPool2d(kernel_size=2)
        elif down == 'conv':
            self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, groups=128)
            self.down4 = nn.Conv2d(256, 256, kernel_size=2, stride=2, groups=256)

            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        if up == 'bilinear' or up == 'nearest':
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
        elif up == 'tconv':
            self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, groups=256)
            self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, groups=128)
            self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, groups=64)
            self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, groups=32)

            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock(n_channel_in, 32, residual, activation)
        self.conv2 = ConvBlock(32, 64, residual, activation)
        self.conv3 = ConvBlock(64, 128, residual, activation)
        self.conv4 = ConvBlock(128, 256, residual, activation)

        self.conv5 = ConvBlock(256, 256, residual, activation)

        self.conv6 = ConvBlock(2 * 256, 128, residual, activation)
        self.conv7 = ConvBlock(2 * 128, 64, residual, activation)
        self.conv8 = ConvBlock(2 * 64, 32, residual, activation)
        self.conv9 = ConvBlock(2 * 32, n_channel_out, residual, activation)

        if self.residual:
            self.convres = ConvBlock(n_channel_in, n_channel_out, residual, activation)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        # print("shapes: c0:%sx:%s c4:%s " % (c0.shape,x.shape,c4.shape))
        x = torch.nn.functional.interpolate(x, size=c4.size()[2:], mode='nearest')
        x = torch.cat([x, c4], 1)  # x[:,0:128]*x[:,128:256],
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.nn.functional.interpolate(x, size=c3.size()[2:], mode='nearest')
        x = torch.cat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.nn.functional.interpolate(x, size=c2.size()[2:], mode='nearest')
        x = torch.cat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.nn.functional.interpolate(x, size=c1.size()[2:], mode='nearest')
        x = torch.cat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        x = self.conv9(x)
        x = torch.nn.functional.interpolate(x, size=c1.size()[2:], mode='nearest')
        if self.residual:
            x = torch.add(x, self.convres(c0))

        return x

class NoiseGenerator2d3d_distributed_ablation(nn.Module):
    def __init__(self, net, unet_opts = 'Unet', device = 'cuda:0', noise_list = 'shot_read_uniform_row1_rowt_periodic'):
        super(NoiseGenerator2d3d_distributed_ablation, self).__init__()
        
        print('generator device', device)
        self.device = device
        self.dtype = torch.float32
        self.noise_list = noise_list
        self.net = net
        self.unet_opts = unet_opts
        self.keep_track = False
        self.all_noise = {}
        
        if 'shot' in noise_list:
            self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002*10000, dtype = self.dtype, device = device), requires_grad = True)
        if 'read' in noise_list:     
            self.read_noise = torch.nn.Parameter(torch.tensor(0.000002*10000, dtype = self.dtype, device = device), requires_grad = True)
        if 'row1' in noise_list:         
            self.row_noise = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device), requires_grad = True)
        if 'rowt' in noise_list:
            self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device), requires_grad = True)
        if 'uniform' in noise_list:    
            self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001*10000, dtype = self.dtype, device = device), requires_grad = True)
        if 'fixed1' in noise_list:
            mean_noise = scipy.io.loadmat(str(_root_dir) + '/data/fixed_pattern_noise.mat')['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
        if 'learnedfixed' in noise_list:
            print('using learned fixed noise')
            
            mean_noise_path = os.environ.get('FIXED_PATTERN_NOISE_MAT', str(_root_dir) + '/data/fixed_pattern_noise.mat')
            mean_noise = scipy.io.loadmat(mean_noise_path)['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
            self.fixednoiset = torch.nn.Parameter(fixednoiset, requires_grad = True)
        
        if 'periodic' in noise_list:
            self.periodic_params = torch.nn.Parameter(torch.tensor([0.0050,0.0050,0.0050], 
                                                                   dtype = self.dtype, device = device)*100, #*1000, 
                                                                  requires_grad = True)
        
        self.indices = None
        
        
        
    def forward(self, x, split_into_patches = False, i0=None):

        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        # pdb.set_trace()
        noise = torch.zeros_like(x)
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            variance = x*self.shot_noise + self.read_noise
            shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise += shot_noise
            if self.keep_track == True:
                self.all_noise['shot_read'] = shot_noise.detach().cpu().numpy() 
        elif 'read' in self.noise_list and 'shot' not in self.noise_list:
            variance =self.read_noise
            noise += torch.randn(x.shape, requires_grad= True, device = self.device)*variance
        elif 'shot' in self.noise_list and 'read' not in self.noise_list:
            variance = x*self.shot_noise
            shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise += shot_noise
        if 'uniform' in self.noise_list:    
            uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
            noise += uniform_noise
            if self.keep_track == True:
                self.all_noise['uniform'] = uniform_noise.detach().cpu().numpy() 
        if 'row1' in self.noise_list: 
            row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)
            noise += row_noise
            if self.keep_track == True:
                self.all_noise['row'] = np.repeat(row_noise.detach().cpu().numpy(), self.all_noise['shot_read'].shape[2], axis=2)
        if 'rowt' in self.noise_list:   
            row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)
            noise += row_noise_temp
        if 'fixed1' in self.noise_list or 'learnedfixed' in self.noise_list:
            if self.indices is not None:
                i1 = self.indices[0]
                i2 = self.indices[1]
            elif i0 is not None:
                i1 = i0[0]
                i2 = i0[1]
            else:
                i1 = np.random.randint(0, self.fixednoiset.shape[-2] - x.shape[-2])
                i2 = np.random.randint(0, self.fixednoiset.shape[-1] - x.shape[-1])
            fixed_noise = self.fixednoiset[...,i1:i1+x.shape[-2], i2:i2 + x.shape[-1]]
            
            noise += fixed_noise
            if self.keep_track == True:
                self.all_noise['fixed'] = fixed_noise.detach().cpu().numpy() 
            
        #elif 'learnedfixed' in self.noise_list:
        #    fixed_noise = self.fixednoiset[...,self.indices[2]:self.indices[3]]
        #    noise += fixed_noise

        if 'periodic' in self.noise_list:
            periodic_noise = torch.zeros(x.shape,  dtype=torch.cfloat, device = self.device)
            periodic_noise[...,0,0] = self.periodic_params[0]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            
            periodic0 = self.periodic_params[1]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device)
            periodic1 = self.periodic_params[2]*torch.randn((x.shape[0:2]),requires_grad= True, device = self.device) 

            periodic_noise[...,0,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            periodic_noise[...,0,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)

            periodic_gen = torch.abs(torch.fft.ifft2(periodic_noise.cpu(), norm="ortho"))
            periodic_gen = periodic_gen.to(self.device)

            noise += periodic_gen
            if self.keep_track == True:
                self.all_noise['periodic'] = periodic_gen.detach().cpu().numpy() 
    
        noisy = x + noise
        
        if split_into_patches== True:
            noisy = split_into_patches2d(noisy)
            x = split_into_patches2d(x)
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy    
    

_reduction_modes = ['none', 'mean', 'sum']

class L0Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, total_iter, loss_weight=1.0, reduction='mean'):
        super(L0Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.total_iter = total_iter

    def forward(self, pred, data, curr_iter, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        gamma = 2.0 * (self.total_iter - curr_iter) / self.total_iter
        return self.loss_weight * (reduce_mean(pred, data)+1e-8) ** gamma
    
import queue

def _init(future_buffer_len): 
    global _global_queue
    _global_queue = queue.Queue()

    global future_buf_len
    future_buf_len = future_buffer_len

    global batch_index
    batch_index = -1

def _clean(): 
    with _global_queue.mutex:
        _global_queue.queue.clear()

def put(value):
    _global_queue.put(value)


def get():
    return _global_queue.get()

def qsize():
    return _global_queue.qsize()

def get_future_buffer_length():
    return future_buf_len
def set_future_buffer_length(future_buffer_len):
    global future_buf_len
    future_buf_len = future_buffer_len
    return future_buf_len


def get_batch_index():
    return batch_index
def set_batch_index(idx):
    global batch_index
    batch_index = idx

def temp_denoise(model, noisyframe, device, args=None):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
    '''

    N, C, H, W = noisyframe.size()
    noisyframe = noisyframe[None, ...]  # Add a batch dimension

    # Perform denoising without using sigma_noise
    out = torch.clamp(model(noisyframe)[0, ...], 0., 1.)  # Clamp the output between 0 and 1

    torch.cuda.synchronize()  # Ensure all CUDA operations are completed

    return out.to(device)  # Return the output on the specified device


def denoise_seq(seq, noise_map, temp_psz, model_temporal, future_buffer_len=0, args=None):
    r"""Denoises a sequence of frames.
    During validation, We use MIMO with global buffer for ease of usage.
    During test, we use pipeline algorithm BSVD with Buffer in each block for better memory-fidelity tradeoff
    Args:
        seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
        noise_map: Tensor. Standard deviation of the added noise
        temp_psz: size of the temporal patch
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """

    device = next(model_temporal.parameters()).device

    numframes, C, H, W = seq.shape
    # For BSVD, we test the video sequence in a single forward
    if temp_psz == -1: temp_psz = numframes 
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    
    num_seg = numframes // temp_psz
    num_last_seg_frames = numframes % temp_psz
    num_batches = num_seg  
    num_batch_frames = temp_psz 
    num_last_batch_frames = numframes % num_batch_frames

    
    _init(future_buffer_len)
    
    for fridx in range(num_batches):
        set_batch_index(fridx)
        start, end = fridx*num_batch_frames, (fridx+1)*num_batch_frames 
        end_new = end + future_buffer_len
        if end_new > numframes:
            end_new = end
            set_future_buffer_length(0)
        inframes = seq[start: end_new]

        #denframes[start: end] = temp_denoise(model_temporal, inframes.to(device), noise_map, seq.device, args=args)[:num_batch_frames] 
        denframes[start: end] = temp_denoise(model_temporal, inframes.to(device), seq.device)[:num_batch_frames]


    set_future_buffer_length(0)
    if num_last_batch_frames > 0:
        if num_last_seg_frames > 0:
            last_sequence = torch.cat((seq[num_seg*temp_psz:], 
                                torch.flip(seq[-(temp_psz-num_last_seg_frames)-1:-1], dims=[0])))
            if num_last_batch_frames == num_last_seg_frames:
                out = temp_denoise(model_temporal, last_sequence.to(device), noise_map, seq.device, args=args)
                denframes[num_seg*temp_psz:] = out[:num_last_seg_frames]
            else:
                inframes = torch.cat((seq[num_batches*num_batch_frames : numframes-num_last_seg_frames],last_sequence))
                out = temp_denoise(model_temporal, inframes.to(device), noise_map, seq.device, args=args)
                denframes[num_batches*num_batch_frames:] = out[:num_last_batch_frames]
            if last_sequence is not None: del last_sequence
        else:
            out = temp_denoise(model_temporal, seq[num_batches*num_batch_frames:].to(device), noise_map, seq.device, args=args)
            denframes[num_batches*num_batch_frames:] = out
        if out is not None: del out  

    # free memory up
    if inframes is not None: del inframes
    
    if noise_map is not None: del noise_map
    
    _clean()
    torch.cuda.empty_cache()
    # convert to appropiate type and return
    return denframes


