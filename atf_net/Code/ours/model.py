import torch
import torch.nn as nn
from torch.nn import functional as F
from Code.lib.res2net_v1b_base import Res2Net_model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

###############################################################################

class MEA0(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(MEA0, self).__init__()
        self.layer_10 = nn.Conv2d(out_dim, out_dim, 1, bias=True)

        n_heads, dropout= 1, 0.3
        self.norm = nn.LayerNorm(out_dim)
        self.attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout)
        self.layer_ful1 = BasicConv2d(out_dim * 2, out_dim, 3, 1, 1)
        

    def forward(self, rgb, depth, flow):
        rgb_proj = self.layer_10(rgb)

        b,c,w,h = rgb.shape
        dep_w = depth.flatten(2).transpose(1, 2)
        flo_w = flow.flatten(2).transpose(1, 2)
        
        attn_out = self.attn(dep_w, flo_w, dep_w)[0]
        attn_out = self.norm(attn_out)
        attn_out = attn_out.view(b,c,w,h)
        
        out = rgb_proj + attn_out
        return out


class MEA(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(MEA, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        
        self.reduc_1 = BasicConv2d(in_dim, out_dim, 1)
        self.reduc_2 = BasicConv2d(in_dim, out_dim, 1)
        self.reduc_3 = BasicConv2d(in_dim, out_dim, 1)
        
        n_heads, dropout= 1, 0.3


        self.attn = nn.MultiheadAttention(out_dim, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

        self.layer_ful1 = BasicConv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_ful2 = BasicConv2d(out_dim + out_dim//2, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth, flow, xx):
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)
        x_flo = self.reduc_3(flow)
        b,c,w,h = x_rgb.shape
               
        dep_w = x_dep
        flo_w = x_flo
        
        dep_w = dep_w.flatten(2).permute(0,2,1)
        flo_w = flo_w.flatten(2).permute(0,2,1)
        
        x11 = self.attn(dep_w, flo_w, dep_w)[0]
        x11 = self.norm(x11)
        
        x11 = x11.view(b,c,w,h)
        x_dep = x_dep.view(b,c,w,h)
        
        x_dep   = x_dep.unsqueeze(1)
        x11   = x11.unsqueeze(1)

        x_cat = torch.cat((x_dep, x11), dim=1)
        
        y0 = x_cat.max(dim=1)[0]

        ful_out = torch.cat((x_rgb, x_rgb.mul(y0)), dim=1)
        out1 = self.layer_ful1(ful_out)
        out2 = self.layer_ful2(torch.cat([out1, xx], dim=1))
         
        return out2

class Attention_block(nn.Module):
    def __init__(self, in_dim):
        super(Attention_block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim*2,in_dim,kernel_size=1)
        act_fn = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
                            nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(in_dim),
                            act_fn,)

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, V_rgb, K_dep, Q_flo):
        b, c, w, h = Q_flo.shape

        Q_flo_1 = Q_flo.flatten(2)
        K_dep_1 = K_dep.flatten(2)
        V_rgb_1 = V_rgb.flatten(2)
        
        T_qvfr = torch.bmm(Q_flo_1.transpose(1, 2), V_rgb_1)
        S_qvfr, T_qvfr_arg = torch.max(T_qvfr, dim=1)
        TA_q = self.bis(Q_flo_1, 2, T_qvfr_arg)
        TA_q = TA_q.view(b, c, w, h)

        S_qvfr = S_qvfr.view(S_qvfr.size(0), 1, w, h)
        C_v = torch.cat([V_rgb, TA_q], 1)
        C_v = self.conv1(C_v)
        C_v = C_v * S_qvfr
        
        T_vkrd = torch.bmm(K_dep_1.transpose(1, 2), V_rgb_1)
        S_vkrd, T_vkrd_arg = torch.max(T_vkrd, dim=1)
        TA_k = self.bis(K_dep_1, 2, T_vkrd_arg)
        TA_k = TA_k.view(b, c, w, h)

        S_vkrd = S_vkrd.view(S_vkrd.size(0), 1, w, h)
        C_k = torch.cat([V_rgb, TA_k], 1)
        C_k = self.conv1(C_k)
        C_k = C_k * S_vkrd

        out = self.conv2(torch.cat([C_v, C_k, V_rgb], 1))
        return out

class MDA(nn.Module):    
    def __init__(self,in_dim):
        super(MDA, self).__init__()
        self.transformer1 = Attention_block(in_dim=in_dim)
        self.transformer2 = Attention_block(in_dim=in_dim)
        self.transformer3 = Attention_block(in_dim=in_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_ful, rgb, dep, flo):
        out1 = self.transformer1(x_ful, rgb, dep)
        out2 = self.transformer2(x_ful, rgb, flo)
        out3 = self.transformer3(rgb, dep, flo)

        out_cat = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1), out3.unsqueeze(1)], dim=1)
        fusion = nn.Softmax(dim=1)(out_cat).max(dim=1)[0]
        out = self.relu(rgb + fusion)
        return out

class ATFNet(nn.Module):
    def __init__(self, channel=32, ind=50):
        super(ATFNet, self).__init__()
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        #Backbone model
        self.layer_rgb  = Res2Net_model(ind)
        self.layer_dep  = Res2Net_model(ind)
        self.layer_flo  = Res2Net_model(ind)

        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        ###############################################
        # funsion encoders #
        ###############################################
        self.mea_0 = MEA0(64, 64)#
        self.mea_1 = MEA(256, 128) 
        self.mea_2 = MEA(512, 256)
        self.mea_3 = MEA(1024, 512)
        self.mea_4 = MEA(2048, 1024)

        self.max_pool = maxpool()
        
        ## rgb
        self.rgb_gcm_4    = BasicConv2d(2048,  channel)
        self.rgb_gcm_3    = BasicConv2d(1024+32,  channel)
        self.rgb_gcm_2    = BasicConv2d(512+32,  channel)
        self.rgb_gcm_1    = BasicConv2d(256+32,  channel)
        self.rgb_gcm_0    = BasicConv2d(64+32,  channel)        
        self.rgb_conv_out = nn.Conv2d(channel, 1, 1)
        
        ## depth
        self.dep_gcm_4    = BasicConv2d(2048,  channel)
        self.dep_gcm_3    = BasicConv2d(1024+32,  channel)
        self.dep_gcm_2    = BasicConv2d(512+32,  channel)
        self.dep_gcm_1    = BasicConv2d(256+32,  channel)
        self.dep_gcm_0    = BasicConv2d(64+32,  channel)        
        self.dep_conv_out = nn.Conv2d(channel, 1, 1)

        ## flow
        self.flo_gcm_4    = BasicConv2d(2048,  channel)
        self.flo_gcm_3    = BasicConv2d(1024+32,  channel)
        self.flo_gcm_2    = BasicConv2d(512+32,  channel)
        self.flo_gcm_1    = BasicConv2d(256+32,  channel)
        self.flo_gcm_0    = BasicConv2d(64+32,  channel)        
        self.flo_conv_out = nn.Conv2d(channel, 1, 1)
        
        ## fusion
        self.ful_gcm_4    = BasicConv2d(1024,  channel)
        self.ful_gcm_3    = BasicConv2d(512+32,  channel)
        self.ful_gcm_2    = BasicConv2d(256+32,  channel)
        self.ful_gcm_1    = BasicConv2d(128+32,  channel)
        self.ful_gcm_0    = BasicConv2d(64+32,  channel)        
        self.ful_conv_out = nn.Conv2d(channel, 1, 1)
        
        self.mda_3   = MDA(channel)
        self.mda_2   = MDA(channel)
        self.mda_1   = MDA(channel)
        self.mda_0   = MDA(channel)
        
    def forward(self, imgs, depths, flows):
        img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(self.layer_dep0(depths))
        flo_0, flo_1, flo_2, flo_3, flo_4 = self.layer_flo(flows)
        
        ful_0 = self.mea_0(img_0, dep_0, flo_0)
        ful_1 = self.mea_1(img_1, dep_1, flo_1, ful_0)
        ful_2 = self.mea_2(img_2, dep_2, flo_2, self.max_pool(ful_1))
        ful_3 = self.mea_3(img_3, dep_3, flo_3, self.max_pool(ful_2))
        ful_4 = self.mea_4(img_4, dep_4, flo_4, self.max_pool(ful_3))
        
        ####################################################
        ## decoder rgb
        ####################################################        
        x_rgb_42    = self.rgb_gcm_4(img_4)
        x_rgb_3_cat = torch.cat([img_3, self.upsample_2(x_rgb_42)], dim=1)
        x_rgb_32    = self.rgb_gcm_3(x_rgb_3_cat)
        x_rgb_2_cat = torch.cat([img_2, self.upsample_2(x_rgb_32)], dim=1)
        x_rgb_22    = self.rgb_gcm_2(x_rgb_2_cat)        
        x_rgb_1_cat = torch.cat([img_1, self.upsample_2(x_rgb_22)], dim=1)
        x_rgb_12    = self.rgb_gcm_1(x_rgb_1_cat)     
        x_rgb_0_cat = torch.cat([img_0, x_rgb_12], dim=1)
        x_rgb_02    = self.rgb_gcm_0(x_rgb_0_cat)     
        rgb_out     = self.upsample_4(self.rgb_conv_out(x_rgb_02))
        ####################################################
        ## decoder depth
        ####################################################        
        x_dep_42    = self.dep_gcm_4(dep_4)
        x_dep_3_cat = torch.cat([dep_3, self.upsample_2(x_dep_42)], dim=1)
        x_dep_32    = self.dep_gcm_3(x_dep_3_cat)
        x_dep_2_cat = torch.cat([dep_2, self.upsample_2(x_dep_32)], dim=1)
        x_dep_22    = self.dep_gcm_2(x_dep_2_cat)        
        x_dep_1_cat = torch.cat([dep_1, self.upsample_2(x_dep_22)], dim=1)
        x_dep_12    = self.dep_gcm_1(x_dep_1_cat)     
        x_dep_0_cat = torch.cat([dep_0, x_dep_12], dim=1)
        x_dep_02    = self.dep_gcm_0(x_dep_0_cat)     
        dep_out     = self.upsample_4(self.dep_conv_out(x_dep_02))
        ####################################################
        ## decoder flow
        ####################################################        
        x_flo_42    = self.flo_gcm_4(flo_4)
        x_flo_3_cat = torch.cat([flo_3, self.upsample_2(x_flo_42)], dim=1)
        x_flo_32    = self.flo_gcm_3(x_flo_3_cat)
        x_flo_2_cat = torch.cat([flo_2, self.upsample_2(x_flo_32)], dim=1)
        x_flo_22    = self.flo_gcm_2(x_flo_2_cat)        
        x_flo_1_cat = torch.cat([flo_1, self.upsample_2(x_flo_22)], dim=1)
        x_flo_12    = self.flo_gcm_1(x_flo_1_cat)     
        x_flo_0_cat = torch.cat([flo_0, x_flo_12], dim=1)
        x_flo_02    = self.flo_gcm_0(x_flo_0_cat)     
        flo_out     = self.upsample_4(self.flo_conv_out(x_flo_02))
        ####################################################
        ## decoder fusion
        ####################################################        
        x_ful_42    = self.ful_gcm_4(ful_4)
        x_ful_3_cat = torch.cat([ful_3, self.upsample_2(self.mda_3(x_ful_42, x_rgb_42, x_flo_42, x_dep_42))], dim=1)
        x_ful_32    = self.ful_gcm_3(x_ful_3_cat)
        x_ful_2_cat = torch.cat([ful_2, self.upsample_2(self.mda_2(x_ful_32, x_rgb_32, x_flo_32, x_dep_32))], dim=1)
        x_ful_22    = self.ful_gcm_2(x_ful_2_cat)        
        x_ful_1_cat = torch.cat([ful_1, self.upsample_2(self.mda_1(x_ful_22, x_rgb_22, x_flo_22, x_dep_22))], dim=1)
        x_ful_12    = self.ful_gcm_1(x_ful_1_cat)     
        x_ful_0_cat = torch.cat([ful_0, self.mda_0(x_ful_12, x_rgb_12, x_flo_12, x_dep_12)], dim=1)
        x_ful_02    = self.ful_gcm_0(x_ful_0_cat)
        ful_out     = self.upsample_4(self.ful_conv_out(x_ful_02))

        return rgb_out, dep_out, flo_out, ful_out