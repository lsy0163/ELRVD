import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

'''
Buffers for memory-based global temporal shift inference
Global queue: past buffer
Future_buffer: future buffer
Copyright @ Junming Chen
'''
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

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, shift_type='TSM', inplace=False, enable_past_buffer=True,
                 **kwargs):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.shift_type = shift_type
        self.inplace = inplace
        self.enable_past_buffer = enable_past_buffer
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if 'TSM' in self.shift_type:
            if self.net.training:
                x = shift(x, self.n_segment, self.shift_type, fold_div=self.fold_div, inplace = self.inplace)
            else:
                x = batch_shift(x, self.shift_type, fold_div=self.fold_div, enable_past_buffer=self.enable_past_buffer)

        return self.net(x)

def shift(x, n_segment, shift_type, fold_div=3, stride=1, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div # 32/8 = 4

    if inplace:
        # Due to some out of order error when performing parallel computing. 
        # May need to write a CUDA kernel.
        print("WARNING: use inplace shift. it has bugs")
        raise NotImplementedError  
        
    else:
        out = torch.zeros_like(x)
        if not 'toFutureOnly' in shift_type:
            out[:, :-stride, :fold] = x[:, stride:, :fold]  # backward (left shift)
            out[:, stride:, fold: 2 * fold] = x[:, :-stride, fold: 2 * fold]  # forward (right shift)
        else:
            out[:, stride:, : 2 * fold] = x[:, :-stride, : 2 * fold] # right shift only
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)


    # Use batch_shift during validating or testing.
def batch_shift(x, shift_type, fold_div=3, stride=1, enable_past_buffer=True):
    nt, c, h, w = x.size()

    fold = c // fold_div
    
    out = torch.zeros_like(x)
    if not 'toFutureOnly' in shift_type: 
        out[:-stride, :fold] = x[stride:, :fold]  # backward (left) shift
        out[stride:, fold: 2 * fold] = x[:-stride, fold: 2 * fold] # forward (right) shift
        
        if enable_past_buffer:
            # memory-based inference
            if get_batch_index() > 0:
                out[:stride, fold: 2 * fold] = get()
            # Keep stride=1, future_buffer_length is abandened
            put(x[-stride-get_future_buffer_length(), fold: 2 * fold])
    else:
        out[stride:, : 2 * fold] = x[:-stride, : 2 * fold] # forward (right) shift only
        
        if enable_past_buffer:
            # memory-based inference
            if get_batch_index() > 0:
                out[:stride, : 2 * fold] = get()
            put(x[-stride-get_future_buffer_length(), : 2 * fold])
    out[:, 2 * fold:] = x[:, 2 * fold:]  # not shift

    
    return out


# if __name__ == '__main__':
#     # test inplace shift v.s. vanilla shift
#     tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
#     tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

#     print('=> Testing CPU...')
#     # test forward
#     with torch.no_grad():
#         for i in range(10):
#             x = torch.rand(2 * 8, 3, 224, 224)
#             y1 = tsm1(x)
#             y2 = tsm2(x)
#             assert torch.norm(y1 - y2).item() < 1e-5

#     # test backward
#     with torch.enable_grad():
#         for i in range(10):
#             x1 = torch.rand(2 * 8, 3, 224, 224)
#             x1.requires_grad_()
#             x2 = x1.clone()
#             y1 = tsm1(x1)
#             y2 = tsm2(x2)
#             grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
#             grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
#             assert torch.norm(grad1 - grad2).item() < 1e-5

#     print('=> Testing GPU...')
#     tsm1.cuda()
#     tsm2.cuda()
#     # test forward
#     with torch.no_grad():
#         for i in range(10):
#             x = torch.rand(2 * 8, 3, 224, 224).cuda()
#             y1 = tsm1(x)
#             y2 = tsm2(x)
#             assert torch.norm(y1 - y2).item() < 1e-5

#     # test backward
#     with torch.enable_grad():
#         for i in range(10):
#             x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
#             x1.requires_grad_()
#             x2 = x1.clone()
#             y1 = tsm1(x1)
#             y2 = tsm2(x2)
#             grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
#             grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
#             assert torch.norm(grad1 - grad2).item() < 1e-5
#     print('Test passed.')


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(CvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        return x

def get_norm_function(norm):
    if norm == "bn":
        norm_fn = nn.BatchNorm2d
    elif norm == "in":
        norm_fn = nn.InstanceNorm2d
    elif norm == 'none':
        norm_fn =nn.Identity
    return norm_fn

def get_act_function(act):
    if act == "relu":
        act_fn = nn.ReLU
    elif act == "relu6":
        act_fn = nn.ReLU6
    elif act == 'none':
        act_fn =nn.Identity
    return act_fn

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', interm_ch = 30, blind=False):
    # def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', blind=False):
        super(InputCvBlock, self).__init__()
        self.interm_ch = interm_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        if blind:
            in_ch = 3
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*in_ch, num_in_frames*self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=bias),
            norm_fn(num_in_frames*self.interm_ch),
            act_fn(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(DownBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True),
            CvBlock(out_ch, out_ch, norm=norm, bias=bias, act=act)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(UpBlock, self).__init__()
        # norm_fn = get_norm_function(norm)
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch, norm=norm, bias=bias, act=act),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(OutputCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=bias),
            norm_fn(in_ch),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', interm_ch=30, blind=False):
    # def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', blind=False):
        super(DenBlock, self).__init__()
        self.chs_lyr0, self.chs_lyr1, self.chs_lyr2 = chns
        
        # if stage2: in_ch=3
        if shift_input:
            self.inc = CvBlock(in_ch=in_ch, out_ch=self.chs_lyr0, norm=norm, bias=bias, act=act)
        else:
            self.inc = InputCvBlock(
                num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, interm_ch=interm_ch, blind=blind)
                # num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, blind=blind)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, norm=norm, bias=bias, act=act)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, norm=norm, bias=bias, act=act)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, norm=norm, bias=bias,    act=act)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, norm=norm, bias=bias,    act=act)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_ch, norm=norm, bias=bias,     act=act)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in1):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(in1)
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)

        # Residual
        x[:, :3, :, :] = in1[:, :3, :, :] - x[:, :3, :, :]

        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.temp1 = DenBlock()

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.temp1(x)
        return x

class WNet_equal(nn.Module):
    def __init__(self, chns=[32, 64, 128], mid_ch=3, shift_input=False, bias=False, act='relu', blind=False):
        super(WNet_equal, self).__init__()
        self.temp1 = DenBlock(chns=chns, out_ch=mid_ch, shift_input=shift_input, bias=bias, act=act, blind=blind)
        self.temp2 = DenBlock(chns=chns, in_ch=mid_ch, shift_input=shift_input, bias=bias, act=act)

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, debug=False):
        # if debug: x_in = x
        x = self.temp1(x)
        # if debug: x_temp1 = x
        x = self.temp2(x)
        # if debug: x_temp2 = x
        return x

class WNet(nn.Module):
    def __init__(self, chns=[32, 64, 128], mid_ch=3, shift_input=False, stage_num=2, in_ch=4, out_ch=4, norm='bn', act='relu', interm_ch=30, blind=False):
    # def __init__(self, chns=[32, 64, 128], mid_ch=3, shift_input=False, stage_num=2, in_ch=4, out_ch=3, norm='bn', act='relu', blind=False):
        super(WNet, self).__init__()
        
        self.stage_num = stage_num
        self.nets_list = nn.ModuleList()
        for i in np.arange(stage_num):
            if i == 0:
                stage_in_ch = in_ch
            else:
                stage_in_ch = mid_ch
            if i == (stage_num-1):
                stage_out_ch = out_ch
            else:
                stage_out_ch = mid_ch
                
            # self.nets_list.append(DenBlock(chns=chns, out_ch=stage_out_ch, in_ch=stage_in_ch, shift_input=shift_input, norm=norm, act=act, interm_ch=interm_ch))
            
            if i == 0:
                self.nets_list.append(DenBlock(chns=chns, out_ch=stage_out_ch, in_ch=stage_in_ch, shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch))
            else:
                self.nets_list.append(DenBlock(chns=chns, out_ch=stage_out_ch,
                                           in_ch=stage_in_ch, shift_input=shift_input, norm=norm, act=act, interm_ch=interm_ch))
        # self.temp2 = DenBlock(chns=chns, in_ch=mid_ch, shift_input=shift_input)

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, debug=False):
        # if debug: x_in = x
        # x = self.temp1(x)
        for i in np.arange(self.stage_num):
            if debug: x_temp1 = x
            x = self.nets_list[i](x)
        # if debug: x_temp2 = x
        return x

class TSN(nn.Module):
    """
        Temporal-shift denoiser during training
    """
    def __init__(self, 
                 num_segments=3,
                 base_model='WNet_multistage', 
                 shift_type='TSM', 
                 shift_div=8,
                 inplace=False,
                 net2d_opt={},
                 enable_past_buffer=True,
                 **kwargs):

        super(TSN, self).__init__()

        self.reshape = True
        self.num_segments = num_segments
        self.shift_type = shift_type
        self.shift_div = shift_div
        self.base_model_name = base_model
        self.net2d_opt = net2d_opt
        self.enable_past_buffer = enable_past_buffer        
        self.inplace = inplace
        self._prepare_base_model(base_model)

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        ShiftOutputCvBlock = int

        if base_model == 'WNet_multistage':
            self.base_model = WNet(**self.net2d_opt)

        else:
            print('No such model')
            raise NotImplementedError

        if self.shift_type != 'no_temporal_shift':
            #from .temporal_shift_ops.temporal_shift import TemporalShift
            for m in self.base_model.modules():
                if isinstance(m, CvBlock) or isinstance(m, ShiftOutputCvBlock):
                    print('Adding temporal shift... {} {}'.format(m.c1, m.c2))
                    m.c1 = TemporalShift(m.c1, self.num_segments, self.shift_div, self.shift_type,
                                         inplace=self.inplace, enable_past_buffer=self.enable_past_buffer)
                    m.c2 = TemporalShift(m.c2, self.num_segments, self.shift_div, self.shift_type,
                                         inplace=self.inplace, enable_past_buffer=self.enable_past_buffer)

    def forward(self, input, noise_map=None):
        # N, F, C, H, W -> (N*F, C, H, W)
        if noise_map != None:
            input = torch.cat([input, noise_map], dim=2)
        if len(input.shape)==5:
            N, F, C, H, W = input.shape
            model_input = input.reshape(N*F, C, H, W)
        else:
            model_input = input
        base_out = self.base_model(model_input)
        if len(input.shape)==5:
            NF, C, H, W = base_out.shape
            base_out = base_out.reshape(N, F, C, H, W)
        return base_out

def extract_dict(ckpt_state, string_name='base_model.temp1.', replace_name=''):
    m_dict = {}
    for k, v in ckpt_state.items():
        if string_name in k:
            m_dict[k.replace(string_name, replace_name)] = v
    return m_dict
        
def replace_dict(ckpt_state, string_name='base_model.temp1.', replace_name=''):
    m_dict = {}
    for k, v in ckpt_state.items():
        # if string_name in k:
        m_dict[k.replace(string_name, replace_name)] = v
    return m_dict

class ShiftConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        ) -> None:
        super(ShiftConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )
        # import pdb; pdb.set_trace()
        # self.conv.weight    = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        # self.conv.bias      = torch.nn.Parameter(torch.zeros_like(self.conv.bias))
    def forward(self, left_fold_2fold, center, right):
        fold_div = 8
        n, c, h, w = center.size()
        fold = c//fold_div
        # import pdb; pdb.set_trace()
        assert left_fold_2fold.size()[1] == fold
        return  self.conv(torch.cat([ right[:, :fold, :, :],
                                     left_fold_2fold, 
                                     center[:, 2*fold:, :, :]], dim=1))
        # return  self.conv(torch.cat([left[:, fold: 2*fold, :, :], center[:, 2*fold:, :, :], right[:, :fold, :, :]], dim=1))

class BiBufferConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True
        ) -> None:
        super(BiBufferConv, self).__init__()
        self.op = ShiftConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        )
        self.out_channels = out_channels
        self.left_fold_2fold = None
        # self.zero_tensor = None
        self.center = None
        
    def reset(self):
        self.left_fold_2fold = None
        self.center = None
        
    def forward(self, input_right, verbose=False):
        fold_div = 8
        if input_right is not None:
            self.n, self.c, self.h, self.w = input_right.size()
            self.fold = self.c//fold_div
        # Case1: In the start or end stage, the memory is empty
        if self.center is None:
            self.center = input_right
            # if verbose:
            
            if input_right is not None:
                if self.left_fold_2fold is None:
                    # In the start stage, the memory and left tensor is empty

                    self.left_fold_2fold = torch.zeros((self.n, self.fold, self.h, self.w), device=torch.device('cuda'))
                if verbose: print("%f+none+%f = none"%(torch.mean(self.left_fold_2fold), torch.mean(input_right)))
            else:
                # in the end stage, both feed in and memory are empty
                if verbose: print("%f+none+none = none"%(torch.mean(self.left_fold_2fold)))
                # print("self.center is None")
            return None
        # Case2: Center is not None, but input_right is None
        elif input_right is None:
            # In the last procesing stage, center is 0
            output =  self.op(self.left_fold_2fold, self.center, torch.zeros((self.n, self.fold, self.h, self.w), device=torch.device('cuda')))
            if verbose: print("%f+%f+none = %f"%(torch.mean(self.left_fold_2fold), torch.mean(self.center), torch.mean(output)))
        else:
            
            output =  self.op(self.left_fold_2fold, self.center, input_right)
            if verbose: print("%f+%f+%f = %f"%(torch.mean(self.left_fold_2fold), torch.mean(self.center), torch.mean(input_right), torch.mean(output)))
            # if output == 57:
                # a = 1
        self.left_fold_2fold = self.center[:, self.fold:2*self.fold, :, :]
        self.center = input_right
        return output

class MemCvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(MemCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = BiBufferConv(in_ch, out_ch, kernel_size=3,
                            padding=1,bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = BiBufferConv(out_ch, out_ch, kernel_size=3,
                            padding=1,bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)


    def forward(self, x):
        x = self.c1(x)
        if x is not None:
            x = self.b1(x)
            x = self.relu1(x)
        x = self.c2(x)
        if x is not None:
            x = self.b2(x)
            x = self.relu2(x)
        return x
    def load(self, state_dict):
        state_dict = replace_dict(state_dict, 'net.', 'op.conv.')
        self.load_state_dict(state_dict)
    
    def reset(self):
        self.c1.reset()
        self.c2.reset()
    
class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(CvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        return x

def get_norm_function(norm):
    if norm == "bn":
        norm_fn = nn.BatchNorm2d
    elif norm == "in":
        norm_fn = nn.InstanceNorm2d
    elif norm == 'none':
        norm_fn =nn.Identity
    return norm_fn

def get_act_function(act):
    if act == "relu":
        act_fn = nn.ReLU
    elif act == "relu6":
        act_fn = nn.ReLU6
    elif act == 'none':
        act_fn =nn.Identity
    return act_fn

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', interm_ch = 30, blind=False):
        super(InputCvBlock, self).__init__()
        # self.interm_ch = 30
        # if with_sigma: channel_per_frame = 4
        # else: channel_per_frame = 3
        self.interm_ch = interm_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        if blind:
            in_ch = 3
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*in_ch, num_in_frames*self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=bias),
            norm_fn(num_in_frames*self.interm_ch),
            act_fn(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True)
        )

    def forward(self, x):
        if x is not None:
            self.n, self.in_channels, self.h, self.w = x.size()
        if x is None:
            return None
        else:
            return self.convblock(x)
    def load(self, state_dict):
        self.load_state_dict(state_dict)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(DownBlock, self).__init__()
        self.out_channels = out_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True),
        )
        self.memconv = MemCvBlock(out_ch, out_ch, norm=norm, bias=bias, act=act)
    def reset(self):
        self.memconv.reset()
    def forward(self, x):
        if x is not None: 
            self.n, self.in_channels, self.h, self.w = x.size()
            x = self.convblock(x)
        return self.memconv(x)

    def load(self, ckpt_state):
        self.convblock[0].load_state_dict(extract_dict(ckpt_state,string_name='convblock.0.'))
        self.convblock[1].load_state_dict(extract_dict(ckpt_state,string_name='convblock.1.'))
        self.memconv.load(extract_dict(ckpt_state, string_name='convblock.3.'))


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(UpBlock, self).__init__()
        self.memconv = MemCvBlock(in_ch, in_ch, norm=norm, bias=bias, act=act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2)
        )
        self.out_channels = out_ch
    def reset(self):
        self.memconv.reset()
    def forward(self, x):
        # if x is None: return None
        if x is not None:
            self.n, self.in_channels, self.h, self.w = x.size()
        x = self.memconv(x)
        if x is not None:
            x = self.convblock(x)
        return x

    def load(self, ckpt_state):
        self.convblock[0].load_state_dict(extract_dict(ckpt_state,string_name='convblock.1.'))
        self.memconv.load(extract_dict(ckpt_state, string_name='convblock.0.'))
        



class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(OutputCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=bias),
            norm_fn(in_ch),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x):
        if x is None: return None
        if x is not None:
            return self.convblock(x)
    def load(self, state_dict):
        self.load_state_dict(state_dict)
        
class MemSkip(nn.Module):
    def __init__(self):
        super(MemSkip, self).__init__()
        self.mem_list = []
    def push(self, x):
        if x is not None:
            self.mem_list.insert(0,x)
            return 1
        else:
            return 0
    def pop(self, x):
        if x is not None:
            return self.mem_list.pop()
        else:
            return None
            

class DenBlock(nn.Module):
    """ Definition of the denosing block
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', interm_ch=30, blind=False):
        super(DenBlock, self).__init__()
        self.chs_lyr0, self.chs_lyr1, self.chs_lyr2 = chns
        if shift_input:
            self.inc = CvBlock(in_ch=in_ch, out_ch=self.chs_lyr0, norm=norm, bias=bias, act=act)
        else:
            self.inc = InputCvBlock(
                num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, interm_ch=interm_ch, blind=blind)

        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, norm=norm, bias=bias, act=act)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, norm=norm, bias=bias, act=act)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, norm=norm, bias=bias,    act=act)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, norm=norm, bias=bias,    act=act)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_ch, norm=norm, bias=bias,     act=act)
        self.skip1  = MemSkip()
        self.skip2  = MemSkip()
        self.skip3  = MemSkip()
        self.reset_params()
    def reset(self):
        self.downc0.reset()
        self.downc1.reset()
        self.upc2.reset()
        self.upc1.reset()
    def load_from(self, ckpt_state):
        self.inc.load(extract_dict(ckpt_state, string_name='inc.'))
        self.downc0.load(extract_dict(ckpt_state, string_name='downc0.'))
        self.downc1.load(  extract_dict(ckpt_state, string_name='downc1.'))
        self.upc2.load(  extract_dict(ckpt_state, string_name='upc2.'))
        self.upc1.load(  extract_dict(ckpt_state, string_name='upc1.'))
        self.outc.load( extract_dict(ckpt_state, string_name='outc.'))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in1):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        self.skip1.push(self.non_slice(in1))
        x0 = self.inc(in1)
        self.skip2.push(x0)
        # Downsampling
        x1 = self.downc0(x0)
        self.skip3.push(x1)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(self.none_add(x2, self.skip3.pop(x2)))
        # Estimation
        x = self.outc(self.none_add(x1, self.skip2.pop(x1)))

        # Residual
        x = self.none_minus(self.skip1.pop(x), x)

        return x
    def non_slice(self, x):
        if x is None:
            return None
        else:
            return x[:, 0:3, :, :]
    def none_add(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            return x1+x2
        
    def none_minus(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            x_out = x2
            x_out[:, :3, :, :] = x1[:, :3, :, :] - x_out[:, :3, :, :]
            return x_out
        

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.temp1 = DenBlock()

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.temp1(x)
        return x

class BSVD(nn.Module):
    """
        Bidirection-buffer based framework with pipeline-style inference
    """
    def __init__(self, chns=[32, 64, 128], mid_ch=3, shift_input=False, in_ch=4, out_ch=4, norm='bn', act='relu', interm_ch=30, blind=False, 
                 pretrain_ckpt='./experiments/pretrained_ckpt/bsvd-64.pth'):
        super(BSVD, self).__init__()
        self.temp1 = DenBlock(chns=chns, out_ch=mid_ch, in_ch=in_ch,  shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch)
        self.temp2 = DenBlock(chns=chns, out_ch=out_ch, in_ch=mid_ch, shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch)

        self.shift_num = self.count_shift()
        # Init weights
        self.reset_params()
        if pretrain_ckpt is not None:
            self.load(pretrain_ckpt)
        # self.shift_num = 
        # self.shift_num = 
    def reset(self):
        self.temp1.reset()
        self.temp2.reset()
    def load(self, path):
        ckpt = torch.load(path)
        print("load from %s"%path)
        #ckpt_state = ckpt['params']
        ckpt_state = ckpt.state_dict()
        # split the dict here
        if 'module' in list(ckpt_state.keys())[0]:
            base_name = 'module.base_model.'
        else:
            base_name = 'base_model.'
        ckpt_state_1 = extract_dict(ckpt_state, string_name=base_name+'nets_list.0.')
        ckpt_state_2 = extract_dict(ckpt_state, string_name=base_name+'nets_list.1.')
        self.temp1.load_from(ckpt_state_1)
        self.temp2.load_from(ckpt_state_2)
            
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def feedin_one_element(self, x):
        x   = self.temp1(x)
        x   = self.temp2(x)
        return x
    
    def forward(self, input, noise_map=None):
        # N, F, C, H, W -> (N*F, C, H, W)
        if noise_map != None:
            input = torch.cat([input, noise_map], dim=2)
        N, F, C, H, W = input.shape
        input = input.reshape(N*F, C, H, W)
        base_out = self.streaming_forward(input)
        NF, C, H, W = base_out.shape
        base_out = base_out.reshape(N, F, C, H, W)
        return base_out
    
    def streaming_forward(self, input_seq):
        """
        pipeline-style inference

        Args:
            Noisy video stream

        Returns:
            Denoised video stream
        """
        out_seq = []
        if isinstance(input_seq, torch.Tensor):
            n,c,h,w = input_seq.shape
            input_seq = [input_seq[i:i+1, ...] for i in np.arange(n)]
        assert type(input_seq) == list, "convert the input into a sequence"
        _,c,h,w = input_seq[0].shape
        with torch.no_grad():
            for i, x in enumerate(input_seq):
                # print("feed in %d image"%i)
                x_cuda = x.cuda()
                x_cuda = self.feedin_one_element(x_cuda)
                # if x_cuda is not None: x_cuda = x_cuda.cpu()
                if isinstance(x_cuda, torch.Tensor):
                    out_seq.append(x_cuda)
                else:
                    out_seq.append(x_cuda)
                # max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
                # print("max memory required \t\t %.2fGB"%max_mem)
                # print("*****************************************************************************")
            end_out = self.feedin_one_element(None)
            # if end_out is not None: end_out = end_out.cpu()
            # if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()
            out_seq.append(end_out)
            # end_out = self.feedin_one_element(0)
            # end stage
            while 1:
                # print("feed in none")
                end_out = self.feedin_one_element(None)
                # if end_out is not None: end_out = end_out.cpu()
                
                if len(out_seq) == (self.shift_num+len(input_seq)):
                    break
                # if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()
                out_seq.append(end_out)
                # max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
                # print("max memory required \t\t %.2fGB"%max_mem)
                # print("*****************************************************************************")
            # number of temporal shift is 2, last element is 0
            # TODO fix init and end frames
            out_seq_clip = out_seq[self.shift_num:]
            self.reset()
            return torch.cat(out_seq_clip, dim=0)

    def count_shift(self):
        count = 0
        for name, module in self.named_modules():
            # print(type(module))
            if "BiBufferConv" in str(type(module)):
                count+=1
        return count

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

cfa = np.array(
    [[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0.65, 0.2784, -0.2784, -0.65], [-0.2784, 0.65, -0.65, 0.2764]])

cfa = np.expand_dims(cfa, axis=2)
cfa = np.expand_dims(cfa, axis=3)
cfa = torch.tensor(cfa).float()  # .cuda()
cfa_inv = cfa.transpose(0, 1)

# dwt dec
h0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
h1 = np.array([-1 / math.sqrt(2), 1 / math.sqrt(2)])
h0 = np.array(h0[::-1]).ravel()
h1 = np.array(h1[::-1]).ravel()
h0 = torch.tensor(h0).float().reshape((1, 1, -1))
h1 = torch.tensor(h1).float().reshape((1, 1, -1))
h0_col = h0.reshape((1, 1, -1, 1))  # col lowpass
h1_col = h1.reshape((1, 1, -1, 1))  # col highpass
h0_row = h0.reshape((1, 1, 1, -1))  # row lowpass
h1_row = h1.reshape((1, 1, 1, -1))  # row highpass
ll_filt = torch.cat([h0_row, h1_row], dim=0)

# dwt rec
g0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
g1 = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
g0 = np.array(g0).ravel()
g1 = np.array(g1).ravel()
g0 = torch.tensor(g0).float().reshape((1, 1, -1))
g1 = torch.tensor(g1).float().reshape((1, 1, -1))
g0_col = g0.reshape((1, 1, -1, 1))
g1_col = g1.reshape((1, 1, -1, 1))
g0_row = g0.reshape((1, 1, 1, -1))
g1_row = g1.reshape((1, 1, 1, -1))


class ColorTransfer(nn.Module):
    def __init__(self):
        super(ColorTransfer, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(cfa)

    def forward(self, x):
        out = self.net1(x)
        return out


class ColorTransferInv(nn.Module):
    def __init__(self):
        super(ColorTransferInv, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(cfa_inv)

    def forward(self, x):
        out = self.net1(x)
        return out


class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        self.net1 = nn.Conv2d(1, 2, kernel_size=(1, 2), stride=(1, 2), padding=0,
                              bias=None)  # Cin = 1, Cout = 4, kernel_size = (1,2)
        self.net1.weight = torch.nn.Parameter(ll_filt)  # torch.Size([2, 1, 1, 2])

    def forward(self, x):
        B, C, H, W = x.shape
        ll = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)
        hl = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)
        lh = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)
        hh = torch.ones([B, 4, int(H / 2), int(W / 2)], device=device)

        for i in range(C):
            ll_ = self.net1(x[:, i:(i + 1) * 1, :, :])  # 1 * 2 * 128 * 64
            y = []
            for j in range(2):
                weight = self.net1.weight.transpose(2, 3)
                y_out = F.conv2d(ll_[:, j:(j + 1) * 1, :, :], weight, stride=(2, 1), padding=0, bias=None)
                y.append(y_out)
            y_ = torch.cat([y[0], y[1]], dim=1)
            ll[:, i:(i + 1), :, :] = y_[:, 0:1, :, :]
            hl[:, i:(i + 1), :, :] = y_[:, 1:2, :, :]
            lh[:, i:(i + 1), :, :] = y_[:, 2:3, :, :]
            hh[:, i:(i + 1), :, :] = y_[:, 3:4, :, :]

        out = torch.cat([ll, hl, lh, hh], dim=1)
        return out


class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(g0_col)  # torch.Size([1,1,2,1])
        self.net2 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=None)
        self.net2.weight = torch.nn.Parameter(g1_col)  # torch.Size([1,1,2,1])

    def forward(self, x):
        lls = x[:, 0:4, :, :]
        hls = x[:, 4:8, :, :]
        lhs = x[:, 8:12, :, :]
        hhs = x[:, 12:16, :, :]
        B, C, H, W = lls.shape
        out = torch.ones([B, C, int(H * 2), int(W * 2)], device=device)
        for i in range(C):
            ll = lls[:, i:i + 1, :, :]
            hl = hls[:, i:i + 1, :, :]
            lh = lhs[:, i:i + 1, :, :]
            hh = hhs[:, i:i + 1, :, :]

            lo = self.net1(ll) + self.net2(hl)  # 1 * 1 * 128 * 64
            hi = self.net1(lh) + self.net2(hh)  # 1 * 1 * 128 * 64
            weight_l = self.net1.weight.transpose(2, 3)
            weight_h = self.net2.weight.transpose(2, 3)
            l = F.conv_transpose2d(lo, weight_l, stride=(1, 2), padding=0, bias=None)
            h = F.conv_transpose2d(hi, weight_h, stride=(1, 2), padding=0, bias=None)
            out[:, i:i + 1, :, :] = l + h
        return out


class Fusion_down(nn.Module):
    def __init__(self):
        super(Fusion_down, self).__init__()
        self.net1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class Fusion_up(nn.Module):
    def __init__(self):
        super(Fusion_up, self).__init__()
        self.net1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class Denoise_down(nn.Module):

    def __init__(self):
        super(Denoise_down, self).__init__()
        self.net1 = nn.Conv2d(21, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = self.net3(net2)
        return out


class Denoise_up(nn.Module):

    def __init__(self):
        super(Denoise_up, self).__init__()
        self.net1 = nn.Conv2d(25, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = self.net3(net2)
        return out


class Refine(nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.net1 = nn.Conv2d(33, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class VideoDenoise(nn.Module):
    def __init__(self):
        super(VideoDenoise, self).__init__()

        self.fusion = Fusion_down()
        self.denoise = Denoise_down()

    def forward(self, ft0, ft1, coeff_a, coeff_b):
        ll0 = ft0[:, 0:4, :, :]
        ll1 = ft1[:, 0:4, :, :]

        # fusion
        sigma_ll1 = torch.clamp(ll1[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)
        fusion_out = torch.mul(ft0, (1 - gamma)) + torch.mul(ft1, gamma)

        # denoise
        sigma_ll0 = torch.clamp(ll0[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, ll1, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)
        return gamma, denoise_out


class MultiVideoDenoise(nn.Module):
    def __init__(self):
        super(MultiVideoDenoise, self).__init__()
        self.fusion = Fusion_up()
        self.denoise = Denoise_up()

    def forward(self, ft0, ft1, gamma_up, denoise_down, coeff_a, coeff_b):
        ll0 = ft0[:, 0:4, :, :]
        ll1 = ft1[:, 0:4, :, :]

        # fusion
        sigma_ll1 = torch.clamp(ll1[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), gamma_up, sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)
        fusion_out = torch.mul(ft0, (1 - gamma)) + torch.mul(ft1, gamma)

        # denoise
        sigma_ll0 = torch.clamp(ll0[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, denoise_down, ll1, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        return gamma, fusion_out, denoise_out, sigma


class MainDenoise(nn.Module): #EMVD
    def __init__(self):
        super(MainDenoise, self).__init__()
        self.ct = ColorTransfer()
        self.cti = ColorTransferInv()
        self.ft = FreTransfer()
        self.fti = FreTransferInv()
        self.vd = VideoDenoise()
        self.md1 = MultiVideoDenoise()
        self.md0 = MultiVideoDenoise()
        self.refine = Refine()

    def transform(self, x):
        net1 = self.ct(x)
        out = self.ft(net1)
        return out

    def transforminv(self, x):
        net1 = self.fti(x)
        out = self.cti(net1)
        return out

    def forward(self, x, coeff_a=1, coeff_b=1):
        ft0 = x[:, 0:4, :, :]  # 1*4*128*128, the t-1 fusion frame
        ft1 = x[:, 4:8, :, :]  # 1*4*128*128, the t frame

        ft0_d0 = self.transform(ft0)        # scale0, torch.Size([1, 16, 256, 256])
        ft1_d0 = self.transform(ft1)

        ft0_d1 = self.ft(ft0_d0[:,0:4,:,:])     # scale1,torch.Size([1, 16, 128, 128])
        ft1_d1 = self.ft(ft1_d0[:, 0:4, :, :])

        ft0_d2 = self.ft(ft0_d1[:,0:4,:,:])     # scale2, torch.Size([1, 16, 64, 64])
        ft1_d2 = self.ft(ft1_d1[:, 0:4, :, :])


        gamma, denoise_out = self.vd(ft0_d2, ft1_d2, coeff_a, coeff_b)
        denoise_out_d2 = self.fti(denoise_out)
        gamma_up_d2 = F.upsample(gamma, scale_factor=2)


        gamma, fusion_out, denoise_out, sigma = self.md1(ft0_d1, ft1_d1, gamma_up_d2, denoise_out_d2, coeff_a, coeff_b)
        denoise_up_d1 = self.fti(denoise_out)
        gamma_up_d1 = F.upsample(gamma, scale_factor=2)

        gamma, fusion_out, denoise_out, sigma = self.md0(ft0_d0, ft1_d0, gamma_up_d1, denoise_up_d1, coeff_a, coeff_b)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out, sigma], axis=1)  # 1 * 36 * 128 * 128
        omega = self.refine(refine_in)  # 1 * 16 * 128 * 128
        refine_out = torch.mul(denoise_out, (1 - omega)) + torch.mul(fusion_out, omega)

        fusion_out = self.transforminv(fusion_out)
        refine_out = self.transforminv(refine_out)
        denoise_out = self.transforminv(denoise_out)

        return gamma, fusion_out, denoise_out, omega, refine_out

"""
Definition of the FastDVDnet model
"""
import torch
import torch.nn as nn

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*4, num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=4 RGGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 64
		self.chs_lyr1 = 128
		self.chs_lyr2 = 256

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=4)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x

class FastDVDnet(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        x: input frames of dim [N, C, H, W], (C=4 RGGB)
    """

    def __init__(self, num_input_frames=5):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        # Define models of each denoising stage
        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)
        # Init weights
        #self.reset_params()
		
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	
    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''
		#pack input frames
        B, N, C, H, W = x.size()
        x = x.view(B, N*C, H, W)

        # Unpack inputs
        (x0, x1, x2, x3, x4) = tuple(x[:, 4*m:4*m+4, :, :] for m in range(self.num_input_frames))

        # First stage
        x20 = self.temp1(x0, x1, x2)
        x21 = self.temp1(x1, x2, x3)
        x22 = self.temp1(x2, x3, x4)

        # Second stage
        x = self.temp2(x20, x21, x22)

        return x
    

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from modules.cbam import CBAM
from modules.dual_attention import CAM_Module, TAM_Module
import functools
from torch.nn import Softmax

class Predenoiser(nn.Module):

    def __init__(self, nf=128):
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

def INF(B, H, W, device):
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device  # Ensure all tensors are on the same device
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # Pass device to INF
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + INF(m_batchsize, height, width, device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        
        return self.gamma * (out_H + out_W) + x


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

class DnCNN_Fusion(nn.Module):
    def __init__(self, channels=4, num_of_layers=17, nframes=3, center=1):
        super(DnCNN_Fusion, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.center = center
        self.nframes = nframes
        layers = []

        layers.append(nn.Conv3d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv3d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv3d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.non_local_attention = Non_Local_Attention(nf=features, nframes=nframes)

        self.fusion_layers = Temporal_Fusion(nf=64, nframes=nframes, center=center)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=features)
        self.recon_trunk = make_layer(ResidualBlock_noBN_end, 10)
        self.cbam = CBAM(features, 16)
        self.conv_last = nn.Conv2d(features, channels, kernel_size, padding=padding)
    
    def forward(self, x):
        B, C, T, H, W = x.size()
        x_center = x[:, self.center, :, :, :].contiguous()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        features = self.dncnn(x)  # Shape: (B, 3, T, H, W)
        features = features.permute(0, 2, 1, 3, 4).contiguous()

        features = self.non_local_attention(features)
        fused_output = self.fusion_layers(features)  # (B, 3, 64, H, W)
        fused_output = self.recon_trunk(fused_output)
        fused_output = self.cbam(fused_output)
        out = self.conv_last(fused_output)
        out = out + x_center
        
        return out

class DnCNN_predenoising(nn.Module): #3DCNN backbone
    def __init__(self, predenoiser, channels=4, num_of_layers=40, nframes=3, center=1):
        super(DnCNN_predenoising, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.center = center
        self.nframes = nframes
        layers = []

        self.pre_denoise = predenoiser

        layers.append(nn.Conv3d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv3d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm3d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv3d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.non_local_attention = Non_Local_Attention(nf=features, nframes=nframes)

        self.fusion_layers = Temporal_Fusion(nf=64, nframes=nframes, center=center)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=features)
        self.recon_trunk = make_layer(ResidualBlock_noBN_end, 10)
        self.cbam = CBAM(features, 16)
        self.conv_last = nn.Conv2d(features, channels, kernel_size, padding=padding)
    
    def forward(self, x):
        B, N, C, H, W = x.size()
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))
        predenoised_img = predenoised_img.view(B, N, C, H, W)

        x_center = x[:, self.center, :, :, :].contiguous()
        predenoised_img = predenoised_img.permute(0, 2, 1, 3, 4).contiguous()
        features = self.dncnn(predenoised_img)  # Shape: (B, 3, T, H, W)
        features = features.permute(0, 2, 1, 3, 4).contiguous()

        features = self.non_local_attention(features)
        fused_output = self.fusion_layers(features)  # (B, 3, 64, H, W)
        fused_output = self.recon_trunk(fused_output)
        fused_output = self.cbam(fused_output)
        out = self.conv_last(fused_output)
        out = out + x_center
        
        return out



if __name__ == '__main__':
    bsvd = BSVD(pretrain_ckpt=None).cuda()
    print(bsvd)
    input = torch.randn(1, 3, 4, 128, 128)
    output = bsvd(input)
    print(output.shape)
