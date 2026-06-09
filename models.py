import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
try:
    from modules.DCNv2_latest.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
#from modules.DCNv4_op.DCNv4.modules.dcnv4 import DCNv4_sep
from modules.cbam import CBAM, ECBAM
from modules.dual_attention import CAM_Module, TAM_Module
#from modules.cc_attention import CrissCrossAttention
from torch.nn import Softmax
from models_util import *

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class ISP(nn.Module):

    def __init__(self):
        super(ISP, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
                  
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
   
        self.upv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv6_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))   
        
        up4 = self.upv4(conv3)
        up4 = torch.cat([up4, conv2], 1)
        conv4 = self.lrelu(self.conv4_1(up4))
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv1], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        conv6 = self.conv6_1(conv5)
        out = nn.functional.pixel_shuffle(conv6, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class Predenoiser(nn.Module):

    def __init__(self, nf=64):
        super(Predenoiser, self).__init__()

        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv8_1 = nn.Conv2d(64, 4, kernel_size=1, stride=1)

    def forward(self, x):

        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv3], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv2], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv1], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        conv8= self.conv8_1(conv7)
        out = conv8
 
        return out

    def lrelu(self, x):
        out = torch.max(0.2*x, x)
        return out

class Alignment(nn.Module):

    def __init__(self, nf=64, groups=1):
        super(Alignment, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised):
        '''align other neighboring frames to the reference frame in the feature level
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l_predenoised[2], ref_fea_l_predenoised[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea_predenoised = self.lrelu(self.L3_dcnpack(nbr_fea_l_predenoised[2], L3_offset))
        L3_fea_noisy = self.lrelu(self.L3_dcnpack(nbr_fea_l_noisy[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l_predenoised[1], ref_fea_l_predenoised[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea_predenoised = self.L2_dcnpack(nbr_fea_l_predenoised[1], L2_offset)
        L3_fea_predenoised = F.interpolate(L3_fea_predenoised, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_predenoised = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea_predenoised, L3_fea_predenoised], dim=1)))
        L2_fea_noisy = self.L2_dcnpack(nbr_fea_l_noisy[1], L2_offset)
        L3_fea_noisy = F.interpolate(L3_fea_noisy, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_noisy = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea_noisy, L3_fea_noisy], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l_predenoised[0], ref_fea_l_predenoised[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea_predenoised = self.L1_dcnpack(nbr_fea_l_predenoised[0], L1_offset)
        L2_fea_predenoised = F.interpolate(L2_fea_predenoised, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea_predenoised = self.L1_fea_conv(torch.cat([L1_fea_predenoised, L2_fea_predenoised], dim=1))
        L1_fea_noisy = self.L1_dcnpack(nbr_fea_l_noisy[0], L1_offset)
        L2_fea_noisy = F.interpolate(L2_fea_noisy, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea_noisy = self.L1_fea_conv(torch.cat([L1_fea_noisy, L2_fea_noisy], dim=1))
        # Cascading
        offset = torch.cat([L1_fea_predenoised, ref_fea_l_predenoised[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea_noisy = self.lrelu(self.cas_dcnpack(L1_fea_noisy, offset))

        return L1_fea_noisy

class Non_Local_Attention(nn.Module):

    def __init__(self, nf=64, nframes=3):
        super(Non_Local_Attention, self).__init__()

        self.conv_before_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())      
        self.conv_before_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_before_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.recurrence = 2
        self.cca = CrissCrossAttention(nf)
        self.ca = CAM_Module()
        self.ta = TAM_Module()

        self.conv_after_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.conv_final = nn.Conv2d(nf, nf, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  

        # spatial non-local attention
        cca_feat = self.conv_before_cca(aligned_fea.reshape(-1, C, H, W))
        for i in range(self.recurrence):
            cca_feat = self.cca(cca_feat)
        cca_conv = self.conv_after_cca(cca_feat).reshape(B, N, C, H, W)

        # channel non-local attention
        ca_feat = self.conv_before_ca(aligned_fea.reshape(-1, C, H, W))
        ca_feat = self.ca(ca_feat)
        ca_conv = self.conv_after_ca(ca_feat).reshape(B, N, C, H, W)

        # temporal non-local attention
        ta_feat = self.conv_before_ta(aligned_fea.permute(0, 2, 1, 3, 4).reshape(-1, N, H, W))
        ta_feat = self.ta(ta_feat)
        ta_conv = self.conv_after_ta(ta_feat).reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4)

        feat_sum = cca_conv+ca_conv+ta_conv
        
        output = self.conv_final(feat_sum.reshape(-1, C, H, W)).reshape(B, N, C, H, W)
                
        return aligned_fea + output


class Temporal_Fusion(nn.Module):

    def __init__(self, nf=64, nframes=3, center=1):
        super(Temporal_Fusion, self).__init__()
        self.center = center

        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nonlocal_fea):
        B, N, C, H, W = nonlocal_fea.size()  

        emb_ref = self.tAtt_2(nonlocal_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(nonlocal_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1) 
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        cor_prob = cor_prob.view(B, -1, H, W)
        nonlocal_fea = nonlocal_fea.view(B, -1, H, W) * cor_prob

        fea = self.lrelu(self.fea_fusion(nonlocal_fea))

        att = self.lrelu(self.sAtt_1(nonlocal_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add

        return fea
    
def make_layer_with_sfb(block, sfb_block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
        layers.append(sfb_block)
    return nn.Sequential(*layers)

class ResidualWithSFB(nn.Module):
    def __init__(self, layer):
        super(ResidualWithSFB, self).__init__()
        self.layer = layer

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out += residual 
        return out

class RViDeNet(nn.Module):
    def __init__(self, predenoiser, nf=16, nframes=3, groups=1, front_RBs=5, back_RBs=10, center=1):
        super(RViDeNet, self).__init__()
        self.center = center

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=nf*4)

        self.pre_denoise = predenoiser

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=groups)

        self.non_local_attention = Non_Local_Attention(nf=nf, nframes=nframes)

        self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=nframes, center=self.center)

        self.recon_trunk = make_layer(ResidualBlock_noBN_end, back_RBs)
        self.cbam = CBAM(nf*4, 16)

        self.conv_last = nn.Conv2d(nf*4, 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # faster version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        x = x.permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)
        predenoised_img = predenoised_img.view(B, N, C, H, W).permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)

        #### extract noisy features
        #print(x[:, :, 0, :, :].contiguous().shape)
        L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
        # L2
        L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
        L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
        # L3
        L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
        L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

        L1_fea_noisy = L1_fea_noisy.view(C*B, N, -1, H, W)
        L2_fea_noisy = L2_fea_noisy.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_noisy = L3_fea_noisy.view(C*B, N, -1, H // 4, W // 4)

        #### extract predenoised features
        L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
        # L2
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
        # L3
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

        L1_fea_predenoised = L1_fea_predenoised.view(C*B, N, -1, H, W)
        L2_fea_predenoised = L2_fea_predenoised.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_predenoised = L3_fea_predenoised.view(C*B, N, -1, H // 4, W // 4)

        #### align
        # ref feature list
        ref_fea_l_noisy = [
            L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
            L3_fea_noisy[:, self.center, :, :, :].clone()
        ]
        ref_fea_l_predenoised = [
            L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
            L3_fea_predenoised[:, self.center, :, :, :].clone()
        ]
        aligned_noisy_fea = []
        for i in range(N):
            nbr_fea_l_noisy = [
                L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                L3_fea_noisy[:, i, :, :, :].clone()
            ]
            nbr_fea_l_predenoised = [
                L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                L3_fea_predenoised[:, i, :, :, :].clone()
            ]
            
            aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
            aligned_noisy_fea.append(aligned_fea_noisy)

        aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
        
        #non-local attention
        non_local_feature = self.non_local_attention(aligned_noisy_fea)

        #temporal fusion
        fea = self.temporal_fusion(non_local_feature)# fea shape: (C*B, nf, H, W)
        _, nf, _, _ = fea.size()
        fusioned_fea_4channel = fea.view(C, B, nf, H, W).permute(1, 0, 2, 3, 4).view(B, C*nf, H, W)
        out = self.recon_trunk(fusioned_fea_4channel)
        out = self.cbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out

class RViDeNet_sfb(nn.Module):
    def __init__(self, predenoiser, nf=16, nframes=3, groups=1, front_RBs=5, back_RBs=10, center=1):
        super(RViDeNet_sfb, self).__init__()
        self.center = center

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=nf*4)

        self.pre_denoise = predenoiser

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=groups)

        self.non_local_attention = Non_Local_Attention(nf=nf, nframes=nframes)

        self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=nframes, center=self.center)
        self.cbam = CBAM(nf*4, 16)

        self.conv_last = nn.Conv2d(nf*4, 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sfb = SFB(embed_dim=nf*4)
        self.recon_trunk = make_layer_with_sfb(ResidualBlock_noBN_end, self.sfb, back_RBs)
        #self.recon_trunk = ResidualWithSFB(self.recon_trunk_init)
        ############################################################

    # faster version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        x = x.permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)
        predenoised_img = predenoised_img.view(B, N, C, H, W).permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)

        #### extract noisy features
        #print(x[:, :, 0, :, :].contiguous().shape)
        L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
        # L2
        L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
        L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
        # L3
        L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
        L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

        L1_fea_noisy = L1_fea_noisy.view(C*B, N, -1, H, W)
        L2_fea_noisy = L2_fea_noisy.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_noisy = L3_fea_noisy.view(C*B, N, -1, H // 4, W // 4)

        #### extract predenoised features
        L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
        # L2
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
        # L3
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

        L1_fea_predenoised = L1_fea_predenoised.view(C*B, N, -1, H, W)
        L2_fea_predenoised = L2_fea_predenoised.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_predenoised = L3_fea_predenoised.view(C*B, N, -1, H // 4, W // 4)

        #### align
        # ref feature list
        ref_fea_l_noisy = [
            L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
            L3_fea_noisy[:, self.center, :, :, :].clone()
        ]
        ref_fea_l_predenoised = [
            L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
            L3_fea_predenoised[:, self.center, :, :, :].clone()
        ]
        aligned_noisy_fea = []
        for i in range(N):
            nbr_fea_l_noisy = [
                L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                L3_fea_noisy[:, i, :, :, :].clone()
            ]
            nbr_fea_l_predenoised = [
                L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                L3_fea_predenoised[:, i, :, :, :].clone()
            ]
            
            aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
            aligned_noisy_fea.append(aligned_fea_noisy)

        aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
        
        #non-local attention
        non_local_feature = self.non_local_attention(aligned_noisy_fea)

        #temporal fusion
        fea = self.temporal_fusion(non_local_feature)# fea shape: (C*B, nf, H, W)
        _, nf, _, _ = fea.size()
        fusioned_fea_4channel = fea.view(C, B, nf, H, W).permute(1, 0, 2, 3, 4).view(B, C*nf, H, W)
        ############################################################################################################
        #cascaded gaze block
        #fusioned_fea_4channel = self.cascaded_gaze_block(fusioned_fea_4channel)
        # for block in self.cascaded_gaze_block:
        #     fusioned_fea_4channel = block(fusioned_fea_4channel)
        #fusioned_fea_4channel = self.net(fusioned_fea_4channel)
        # fusioned_fea_4channel = self.transformer_block1(fusioned_fea_4channel)
        # fusioned_fea_4channel = self.transformer_block2(fusioned_fea_4channel)
        # fusioned_fea_4channel = self.transformer_block3(fusioned_fea_4channel)

        #fusioned_fea_4channel = self.sfb(fusioned_fea_4channel)
        ############################################################################################################

        #spatial fusion
        #fusioned_fea_4channel = fusioned_fea_4channel.unsqueeze(1)
        #print(fusioned_fea_4channel.shape)
        out = self.recon_trunk(fusioned_fea_4channel)
        #out = self.recon_trunk2(out)
        #print(out.shape)
        #out = out.squeeze(1)
        #out = self.conv_before_cbam(out)
        # out = self.transformer_block1(out)
        # out = self.transformer_block2(out)
        # out = self.transformer_block3(out)
        # out = self.transformer_block4(out)
        # out = self.transformer_block5(out)
        out = self.cbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out

class RViDeNet_ECBAM(nn.Module):
    def __init__(self, predenoiser, nf=16, nframes=3, groups=1, front_RBs=5, back_RBs=10, center=1):
        super(RViDeNet_ECBAM, self).__init__()
        self.center = center

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=nf*4)

        self.pre_denoise = predenoiser

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=groups)

        self.non_local_attention = Non_Local_Attention(nf=nf, nframes=nframes)

        self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=nframes, center=self.center)

        self.recon_trunk = make_layer(ResidualBlock_noBN_end, back_RBs)
  
        #self.cbam = CBAM(nf*4, 16)
        self.ecbam = ECBAM(nf*4, 16)

        self.conv_last = nn.Conv2d(nf*4, 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # faster version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        x = x.permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)
        predenoised_img = predenoised_img.view(B, N, C, H, W).permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)

        #### extract noisy features
        #print(x[:, :, 0, :, :].contiguous().shape)
        L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
        # L2
        L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
        L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
        # L3
        L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
        L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

        L1_fea_noisy = L1_fea_noisy.view(C*B, N, -1, H, W)
        L2_fea_noisy = L2_fea_noisy.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_noisy = L3_fea_noisy.view(C*B, N, -1, H // 4, W // 4)

        #### extract predenoised features
        L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
        # L2
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
        # L3
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

        L1_fea_predenoised = L1_fea_predenoised.view(C*B, N, -1, H, W)
        L2_fea_predenoised = L2_fea_predenoised.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_predenoised = L3_fea_predenoised.view(C*B, N, -1, H // 4, W // 4)

        #### align
        # ref feature list
        ref_fea_l_noisy = [
            L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
            L3_fea_noisy[:, self.center, :, :, :].clone()
        ]
        ref_fea_l_predenoised = [
            L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
            L3_fea_predenoised[:, self.center, :, :, :].clone()
        ]
        aligned_noisy_fea = []
        for i in range(N):
            nbr_fea_l_noisy = [
                L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                L3_fea_noisy[:, i, :, :, :].clone()
            ]
            nbr_fea_l_predenoised = [
                L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                L3_fea_predenoised[:, i, :, :, :].clone()
            ]
            
            aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
            aligned_noisy_fea.append(aligned_fea_noisy)

        aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
        
        #non-local attention
        non_local_feature = self.non_local_attention(aligned_noisy_fea)

        #temporal fusion
        fea = self.temporal_fusion(non_local_feature)# fea shape: (C*B, nf, H, W)
        _, nf, _, _ = fea.size()
        fusioned_fea_4channel = fea.view(C, B, nf, H, W).permute(1, 0, 2, 3, 4).view(B, C*nf, H, W)
        out = self.recon_trunk(fusioned_fea_4channel)
        #out = self.cbam(out)
        out = self.ecbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out

    '''# old version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        ### predenoising
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        aligned_noisy_fea_4channel = []
        fusioned_fea_4channel = []

        for channel_index in range(C):

            #### extract noisy features
            L1_fea_noisy = self.lrelu(self.conv_first(x[:,:,channel_index,:,:].view(-1, 1, H, W)))
            L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
            # L2
            L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
            L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
            # L3
            L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
            L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

            L1_fea_noisy = L1_fea_noisy.view(B, N, -1, H, W)
            L2_fea_noisy = L2_fea_noisy.view(B, N, -1, H // 2, W // 2)
            L3_fea_noisy = L3_fea_noisy.view(B, N, -1, H // 4, W // 4)

            #### extract predenoised features
            L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:,channel_index,:,:].view(-1, 1, H, W)))
            L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
            # L2
            L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
            L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
            # L3
            L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
            L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

            L1_fea_predenoised = L1_fea_predenoised.view(B, N, -1, H, W)
            L2_fea_predenoised = L2_fea_predenoised.view(B, N, -1, H // 2, W // 2)
            L3_fea_predenoised = L3_fea_predenoised.view(B, N, -1, H // 4, W // 4)

            #### align
            # ref feature list
            ref_fea_l_noisy = [
                L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
                L3_fea_noisy[:, self.center, :, :, :].clone()
            ]
            ref_fea_l_predenoised = [
                L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
                L3_fea_predenoised[:, self.center, :, :, :].clone()
            ]
            aligned_noisy_fea = []
            for i in range(N):
                nbr_fea_l_noisy = [
                    L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                    L3_fea_noisy[:, i, :, :, :].clone()
                ]
                nbr_fea_l_predenoised = [
                    L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                    L3_fea_predenoised[:, i, :, :, :].clone()
                ]
                
                aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
                aligned_noisy_fea.append(aligned_fea_noisy)

            aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
            aligned_noisy_fea_4channel.append(aligned_noisy_fea) 
            
            #non-local attention
            non_local_feature = self.non_local_attention(aligned_noisy_fea)

            #temporal fusion
            fea = self.temporal_fusion(non_local_feature)
            fusioned_fea_4channel.append(fea)

        aligned_noisy_fea_4channel = torch.cat(aligned_noisy_fea_4channel, dim=2) 
        fusioned_fea_4channel = torch.cat(fusioned_fea_4channel, dim=1)
        
        #spatial fusion
        out = self.recon_trunk(fusioned_fea_4channel)
        out = self.cbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out'''

class NAFNet(nn.Module):
    def __init__(self, img_channel=4, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
class CascadedGaze(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=10, enc_blk_nums=[2, 2, 4, 6], dec_blk_nums=[2, 2, 2, 2], GCE_CONVS_nums=[3,3,2,2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        # for num in enc_blk_nums:
        for i in range(len(enc_blk_nums)):
            num = enc_blk_nums[i]
            GCE_Convs = GCE_CONVS_nums[i]
            self.encoders.append(
                nn.Sequential(
                    *[CascadedGazeBlock(chan, GCE_Conv=GCE_Convs) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock0(chan) for _ in range(middle_blk_num)]
            )

        for i in range(len(dec_blk_nums)):
            num = dec_blk_nums[i]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock0(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    

## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=4, 
        out_channels=4, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

class RViDeNet_(nn.Module):
    def __init__(self, predenoiser, nf=64, nframes=3, groups=1, front_RBs=10, back_RBs=10, center=1):
        super(RViDeNet_, self).__init__()
        self.center = center

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=nf*4)

        self.pre_denoise = predenoiser

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=groups)

        self.non_local_attention = Non_Local_Attention(nf=nf, nframes=nframes)

        self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=nframes, center=self.center)

        self.recon_trunk = make_layer(ResidualBlock_noBN_end, back_RBs)
    
        self.cbam = CBAM(nf*4, 16)

        self.conv_last = nn.Conv2d(nf*4, 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # faster version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        x = x.permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)
        predenoised_img = predenoised_img.view(B, N, C, H, W).permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)

        #### extract noisy features
        L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
        # L2
        L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
        L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
        # L3
        L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
        L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

        L1_fea_noisy = L1_fea_noisy.view(C*B, N, -1, H, W)
        L2_fea_noisy = L2_fea_noisy.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_noisy = L3_fea_noisy.view(C*B, N, -1, H // 4, W // 4)

        #### extract predenoised features
        L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
        # L2
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
        # L3
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

        L1_fea_predenoised = L1_fea_predenoised.view(C*B, N, -1, H, W)
        L2_fea_predenoised = L2_fea_predenoised.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_predenoised = L3_fea_predenoised.view(C*B, N, -1, H // 4, W // 4)

        #### align
        # ref feature list
        ref_fea_l_noisy = [
            L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
            L3_fea_noisy[:, self.center, :, :, :].clone()
        ]
        ref_fea_l_predenoised = [
            L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
            L3_fea_predenoised[:, self.center, :, :, :].clone()
        ]
        aligned_noisy_fea = []
        for i in range(N):
            nbr_fea_l_noisy = [
                L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                L3_fea_noisy[:, i, :, :, :].clone()
            ]
            nbr_fea_l_predenoised = [
                L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                L3_fea_predenoised[:, i, :, :, :].clone()
            ]
            
            aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
            aligned_noisy_fea.append(aligned_fea_noisy)

        aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
        
        #non-local attention
        non_local_feature = self.non_local_attention(aligned_noisy_fea)

        #temporal fusion
        fea = self.temporal_fusion(non_local_feature)# fea shape: (C*B, nf, H, W)
        _, nf, _, _ = fea.size()
        fusioned_fea_4channel = fea.view(C, B, nf, H, W).permute(1, 0, 2, 3, 4).view(B, C*nf, H, W)
        out = self.recon_trunk(fusioned_fea_4channel)
        out = self.cbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out
    
# from torchsummaryX import summary

# if __name__ == '__main__':
#     predenoiser = Predenoiser().cuda()  
#     model = RViDeNet_(predenoiser=predenoiser).cuda()  

#     # Create input tensor
#     input_tensor = torch.randn(1, 3, 4, 128, 128).cuda()

#     print("Model Summary")
#     summary(model, input_tensor)

    

