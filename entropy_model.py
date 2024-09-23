import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F 
from typing import List
_LOG_SCALES_MIN = -7


def log_softmax(logit_weight, dim):
    #可以认为是对logit_weight在维度dim上softmax，但是这样会更稳定
    m,_ = torch.max(logit_weight, dim=dim, keepdim=True)
    return logit_weight - m - torch.log(torch.sum(torch.exp(logit_weight - m),dim=dim, keepdim=True))


def log_sum_exp(log_weight, dim):
    m, _ = torch.max(log_weight, dim=dim)
    m_keep, _ = torch.max(log_weight, dim=dim, keepdim=True)
    return log_weight.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)


#different from compressai, quantization is done before it
class LogisticMixture(nn.Module):
    def __init__(self):
        super().__init__()
    
    def param_prepare(self, x:Tensor, params:Tensor, param_num:int):
        """
        prepare x and parameters for forward function to calculate likelihood
        param_num: int, 3 for s>0 and 4 for s=0
        x:[B,C,H,W]
        params:[B,C*K*param_num,H,W]
        return:x_reshape:[B,C,1,H,W]
               miu:[B,C,K,H,W]
               log_scale:[B,C,K,H,W]
               weight:[B,C,K,H,W]
        """
        B, C, H, W = x.shape
        C_param = params.shape[1]
        K = C_param//C//param_num            
        x_out = x.reshape([B,C,1,H,W])
        if param_num==3:
            miu, log_scale, weight = torch.chunk(params, chunks=3, dim=1)
            #miu:[B,C*K,H,W]
            log_scale = torch.clamp(log_scale, min=_LOG_SCALES_MIN)
            miu = miu.reshape([B,C,K,H,W])
            log_scale = log_scale.reshape([B,C,K,H,W])
            weight = weight.reshape([B,C,K,H,W])
        elif param_num==4:
            assert C==3
            miu, log_scale, weight, lamda= torch.chunk(params,4,dim=1)
            lamda_a, lamda_b, lamda_c = torch.chunk(lamda, 3, dim=1)
            #lamda_a [B,K,H,W]
            miu_r = miu[:,0:K, :, :]
            miu_g = miu[:,K:2*K,:,:]
            miu_b = miu[:,2*K:,:,:]
            miu_g = miu_g + torch.sigmoid(lamda_a)*x[:,0,:,:].unsqueeze(dim=1)
            miu_b = miu_b + torch.sigmoid(lamda_b)*x[:,0,:,:].unsqueeze(dim=1) + torch.sigmoid(lamda_c)*x[:,1,:,:].unsqueeze(dim=1)
            miu = torch.concatenate([miu_r, miu_g, miu_b], dim=1)
            log_scale = torch.clamp(log_scale, min=_LOG_SCALES_MIN)
            miu = miu.reshape([B,C,K,H,W])
            log_scale = log_scale.reshape([B,C,K,H,W])
            weight = weight.reshape([B,C,K,H,W])
        else:
            raise RuntimeError("param_num must be 3 or 4")
        return x_out, miu, log_scale, weight


    def forward(self, quantized_x:Tensor, params:Tensor, param_num:int, bin:float, x_min:float, x_max:float):
        """
        quantized_x:[B,C,H,W] 
        params:[B,C*K*param_num,H,W]entropy parameters, network output
        param_num:3 or 4
        bin:quantization_bin
        x_min:minimum scalar for quantization
        x_max:maximum scalar for quantization
        return log_likelihood[B,C,H,W]
        """
        assert not torch.isnan(params).any()
        x_out, miu, log_scale, weight = self.param_prepare(quantized_x, params, param_num)
        #x_out:[B,C,1,H,W]
        #miu, log_scale, weight:[B,C,K,H,W]
        #下面求的都是每个k对应的logistic分布的log值
        scale = torch.exp(-log_scale)
        plus_in = scale*(x_out + bin/2 - miu)
        min_in = scale*(x_out - bin/2 -miu)
        cdf_plus = torch.sigmoid(plus_in)
        cdf_min = torch.sigmoid(min_in)
        cdf_delta = cdf_plus - cdf_min
        log_cdf_plus = plus_in - F.softplus(plus_in)  #for x_out==x_min; plus_in-F.softmax(plus_in)==log(sigmoid(plus_in))
        log_one_minus_cdf_min = -F.softplus(min_in)  #for x_out==x_max; -F.softmax(min_in)==log(1-sigmoid(min_in))

        x_min = x_min + 1e-4
        x_max = x_max - 1e-4
        #NOTE: three conditions
        #if x < x_min:    cond_C
        #        log_cdf_plus      out_c
        #elif x > x_max:  cond_B
        #        log_one_minus_cdf_min  out_B
        #else
        #        log(cdf_delta)        out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        cond_B = (x_out > x_max).float()
        out_B = (cond_B*log_one_minus_cdf_min + (1. - cond_B)*out_A)
        cond_C = (x_out < x_min).float()
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B
        #log_probs = torch.clamp(log_probs,min=-25, max=25)
        #conbine with pi
        log_probs_weighted = log_probs.add(
            log_softmax(weight, dim=2)
        )

        neg_log_likelihood = -log_sum_exp(log_probs_weighted, dim=2)/math.log(2)
        #neg_log_likelihood = torch.clip(neg_log_likelihood, min=1e-3)
        """
        if torch.isnan(neg_log_likelihood).any():
            print("log_scale.min():",log_scale.min())
            print("log_scale.max():", log_scale.max())
            print("scale.min():",scale.min())
            print("scale.max():",scale.max())
            print("cdf_delta.min():",cdf_delta.min())
            print("cdf_delta.max():",cdf_delta.max())
            exit(-1)
        
        if torch.isinf(neg_log_likelihood).any():
            print("log_scale.min():",log_scale.min())
            print("log_scale.max():", log_scale.max())
            print("scale.min():",scale.min())
            print("scale.max():",scale.max())
            print("cdf_delta.min():",cdf_delta.min())
            print("cdf_delta.max():",cdf_delta.max())
            exit(-1)
        
        if torch.sum(neg_log_likelihood)<10:
            print(log_scale[0,0,:,:])
            print(scale[0,0,0,:,:])
            print(miu[0,0,0,:,:])
            print(x_out[0,0,0,:,:])
            print(log_probs_weighted[0,0,0,:,:])
            print(neg_log_likelihood[0,0,:,:])
        """
        return neg_log_likelihood
    
    @torch.no_grad()
    def param_prepare_cdf_z(self, shape, params:Tensor, param_num:int):
        B, C, H, W = shape
        C_param = params.shape[1]
        K = C_param//C//param_num            
        if param_num==3:
            miu, log_scale, weight = torch.chunk(params, chunks=3, dim=1)
            #miu:[B,C*K,H,W]
            log_scale = torch.clamp(log_scale, min=_LOG_SCALES_MIN)
            miu = miu.reshape([B,C,K,H,W,1])
            log_scale = log_scale.reshape([B,C,K,H,W,1])
            weight = weight.reshape([B,C,K,H,W,1])
        else:
            raise RuntimeError("param_num must be 3 or 4")
        return miu, log_scale, weight
    
    @torch.no_grad()
    def param_prepare_cdf_x(self, shape, channel, decoded:List, params:Tensor):
        B,C,H,W = shape
        assert C==3
        C_param = params.shape[1]
        K = C_param//C//4         
        miu, log_scale, weight, lamda= torch.chunk(params,4,dim=1)
        lamda_a, lamda_b, lamda_c = torch.chunk(lamda, 3, dim=1)
        #for red
        if channel==0:
            miu_channel = miu[:,0:K, :, :]
            log_scale_channel = log_scale[:,0:K,:,:]
            weight_channel = weight[:,0:K,:,:]
        elif channel==1:
            miu_channel = miu[:,K:2*K,:,:] + torch.sigmoid(lamda_a)*decoded[0].unsqueeze(1)
            log_scale_channel = log_scale[:,K:2*K,:,:]
            weight_channel = weight[:,K:2*K,:,:]
        else:
            miu_channel = miu[:,2*K:,:,:] + torch.sigmoid(lamda_b)*decoded[0].unsqueeze(1) + torch.sigmoid(lamda_c)*decoded[1].unsqueeze(1)
            log_scale_channel = log_scale[:,2*K:, :,:]
            weight_channel = weight[:,2*K:,:,:]
        log_scale_channel = torch.clamp(log_scale_channel, min=_LOG_SCALES_MIN)
        miu_channel = miu_channel.reshape([B,1,K,H,W,1])
        log_scale_channel = log_scale_channel.reshape([B,1,K,H,W,1])
        weight_channel = weight_channel.reshape([B,1,K,H,W,1])
        return miu_channel, log_scale_channel, weight_channel
  

    #TODO: 按照cheng_anchor的做法，先算pmf，再clamp, precision=16, 再算cdf
    """
    def get_cdf(
            #TODO:renormalize
            self, 
            shape, 
            miu:Tensor,
            log_scale:Tensor,
            weight:Tensor,
            level_num, 
            bin:float, 
            x_min:float, 
            x_max:float):
        
        calculate cdf using params, level_num
        miu:[B,C,K,H,W]
        log_scale:[B,C,K,H,W]
        weight:[B,C,K,H,W]
        
        B,C,H,W = shape
        K = miu.shape[2]
        samples = torch.arange(0,level_num,1).to(miu.device)
        samples = samples*(x_max-x_min)/(level_num-1) + x_min
        samples = samples.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([B,C,K,H,W,level_num])
        scale = torch.exp(-log_scale)
        plus_in = scale*(samples + bin/2 - miu)
        min_in = scale*(samples - bin/2 -miu)
        cdf_plus = torch.sigmoid(plus_in)
        cdf_min = torch.sigmoid(min_in)
        cdf_delta = cdf_plus - cdf_min
        log_cdf_plus = plus_in -F.softplus(plus_in)  #for x_out==x_min; plus_in-F.softmax(plus_in)==log(sigmoid(plus_in))
        log_one_minus_cdf_min = -F.softplus(min_in)  #for x_out==x_max; -F.softmax(min_in)==log(1-sigmoid(min_in))

        x_min = x_min + 1e-5
        x_max = x_max - 1e-5
        #NOTE: three conditions
        #if x < x_min:    cond_C
        #        log_cdf_plus      out_c
        #elif x > x_max:  cond_B
        #        log_one_minus_cdf_min  out_B
        #else
        #        log(cdf_delta)        out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        cond_B = (samples > x_max).float()
        out_B = (cond_B*log_one_minus_cdf_min + (1. - cond_B)*out_A)
        cond_C = (samples < x_min).float()
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B

        #conbine with pi
        log_probs_weighted = log_probs.add(
            log_softmax(weight, dim=2)
        )
        pmf = torch.exp(log_probs_weighted).sum(dim=2)   #pmf[B,C,H,W,L]
        
        #clamp
        pmf = pmf.clamp(min=1.0/65536, max=1.0)
        pmf = pmf/torch.sum(pmf, dim=-1, keepdim=True)
        cdf = torch.cumsum(pmf, dim=-1)
        zero = torch.zeros([B,C,H,W,1]).to(pmf.device)
        cdf = torch.concatenate((zero, cdf),dim=-1).to(torch.float32)
        cdf = cdf.clamp(min=0,max=1.0)
        return cdf
        """
    
    def get_cdf(
            #TODO:renormalize
            self, 
            shape, 
            miu:Tensor,
            log_scale:Tensor,
            weight:Tensor,
            level_num, 
            bin:float, 
            x_min:float, 
            x_max:float):
        
        B,C,H,W = shape
        K = miu.shape[2]
        samples = torch.arange(0,level_num,1).to(miu.device)
        samples = samples*(x_max-x_min)/(level_num-1) + x_min
        scale = torch.exp(-log_scale)
        pi = F.softmax(weight,2)
        cdf = torch.sum(pi*torch.sigmoid(scale*(samples-miu+bin/2)),dim=2)
        cdf_zeros = torch.zeros((B,C,H,W,1)).to(miu.device)
        cdf = torch.concatenate((cdf_zeros, cdf),dim=-1)
        cdf[:,:,:,:,level_num] = 1
        pmf = cdf[:,:,:,:,1:] - cdf[:,:,:,:,0:level_num]

        pmf = pmf.clamp(min=1.0/65536, max=1.0)
        pmf = pmf/torch.sum(pmf, dim=-1, keepdim=True)
        cdf = torch.cumsum(pmf, dim=-1)
        zero = torch.zeros([B,C,H,W,1]).to(pmf.device)
        cdf = torch.concatenate((zero, cdf),dim=-1).to(torch.float32)
        cdf = cdf.clamp(min=0,max=1.0)
        return cdf


        

class UniformDistribution(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, quantized_x:Tensor, level):
        return -torch.log(torch.zeros_like(quantized_x).fill_(1/level))/math.log(2)
    
    @torch.no_grad()
    def get_cdf(self, shape:Tensor, device, level):
        B,C,H,W = shape
        pmf = torch.zeros([B,C,H,W,level]).to(device)
        pmf = torch.fill(pmf, 1/level).clamp(min=1.0/65536, max=1.0)
        pmf = pmf/torch.sum(pmf,dim=-1,keepdim=True)
        cdf = torch.cumsum(pmf, dim=-1)
        zero = torch.zeros([B,C,H,W,1]).to(pmf.device)
        cdf = torch.concatenate((zero, cdf),dim=-1).to(torch.float32)
        cdf = cdf.clamp(min=0, max=1)
        return cdf