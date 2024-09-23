import torch
import torch.nn as nn
from L3C import*
from torch import Tensor
from entropy_model import LogisticMixture, UniformDistribution
from ptflops import get_model_complexity_info 
import torchac
import random


class NetConfig():
    def __init__(
            self, 
            channel_num=64,
            res_num=8,
            C_Q = 5,
            K=10,
            layer = 3,
            z_min = -1,
            z_max = 1,
            z_level = 25,
            z_bin = 1/12,
            softness = 2,
    ):
        self.ch = channel_num
        self.res_num = res_num
        self.C_Q = C_Q
        self.K = int(K)
        self.layer = layer
        self.z_min = z_min
        self.z_max = z_max
        self.z_level = z_level
        self.z_bin = z_bin
        self.softness = softness


def bitencode(sym, cdf):
    symbol = sym.to(torch.int16).to("cpu")
    cdf_cpu = cdf.to("cpu")
    string = torchac.encode_float_cdf(cdf_cpu, symbol, check_input_bounds=True)
    return string


def bitdecode(cdf, string):
    cdf_cpu = cdf.to("cpu")
    symbol = torchac.decode_float_cdf(cdf_cpu, string)
    return symbol
    """
    levels = torch.arange(0, level, 1)
    levels = levels*(max-min)/(level-1) + min
    dequantized_z = torch.take(levels, symbol).to(cdf.device)
    """
    return dequantized_z

def stable_softmax(x:Tensor, dim:int)->Tensor:
    assert not torch.isnan(x).any()
    max_value,_ = torch.max(x, dim=dim, keepdim=True)
    numerator = torch.exp(x-max_value)
    denominator = torch.sum(torch.exp(x-max_value),dim=dim, keepdim=True)
    return numerator/denominator


class Quantizer():
    def quantize(self, x:Tensor, min, max, level, mode="soft", softness=None):
        """
        x:tensor to quantize
        min,max:scalar scale
        level:scalar number
        bin: quantization bin
        mode: "soft" or "hard"
        """
        B, C, H, W = x.shape
        levels = torch.arange(0,level,1).to(x.device)
        levels = levels*(max-min)/(level-1) + min
        levels_expand = levels.unsqueeze(0)
        levels_expand = levels_expand.unsqueeze(0)
        levels_expand = levels_expand.unsqueeze(0)
        levels_expand = levels_expand.unsqueeze(0)
        #levels_expand[1,1,1,1,L]
        levels_expand = levels_expand.expand([B,C,H,W,level])
        x_expand = x.unsqueeze(-1).expand([B,C,H,W,level])
        dist = torch.pow((x_expand-levels_expand),2)
        if mode=="soft":
            if softness==None:
                raise RuntimeError("softness is None")
            soft_dist = softness * dist
            dist_softmax = torch.softmax(-soft_dist,dim=-1)
            dist_stable_softmax = stable_softmax(soft_dist,dim=-1)
            #assert not torch.isnan(dist_stable_softmax).any()
            #assert not torch.isnan(dist_softmax).any()
            quantized_x = torch.sum(torch.softmax(-soft_dist,dim=-1)*levels_expand, dim=-1)   #B,C,H,W
            
            x_index = torch.argmin(dist.detach(), dim=-1)    #B,C,H,W
            levels = torch.arange(0,level,1).to(x.device)
            levels = levels*(max-min)/(level-1) + min
            quantized_data = torch.take(levels, x_index)  #B,C,H,W
            quantized_x.data = quantized_data.data
            assert not torch.isnan(quantized_x).any()
            assert not torch.isinf(quantized_x).any()

        elif mode=="hard":
            x_index = torch.argmin(dist.detach(), dim=-1)    #B,C,H,W
            levels = torch.arange(0,level,1).to(x.device)
            levels = levels*(max-min)/(level-1) + min
            quantized_x = torch.take(levels, x_index)  #B,C,H,W
        elif mode=="symbol":
            quantized_x = torch.argmin(dist.detach(), dim=-1).to(torch.int16)
        else:
            raise NotImplementedError("choose mode in soft, hard and symbol")
        return quantized_x
    

    def dequantize(self, index:Tensor, min:float, max:float, level):
        """
        index:symbol
        """
        levels = torch.arange(0,level,1).to(index.device)
        levels = levels*(max-min)/(level-1) + min
        dequantized = torch.take(levels, index)
        return dequantized


class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = args.layer
        self.z_min = args.z_min
        self.z_max = args.z_max
        self.z_level = args.z_level
        self.softness = args.softness
        self.z_bin = args.z_bin
        self.K = args.K

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.meanshift = MeanShift(255., rgb_mean=rgb_mean, rgb_std=rgb_std)
        self.head = nn.Sequential(MeanShift(0,(0.,0.,0.),(128.,128.,128.)), conv3x3(in_ch=3, out_ch=args.ch))
        self.encoders = nn.ModuleList()
        for i in range(self.layers-1):
            self.encoders.append(
                EncUnit(in_ch=args.ch, out_ch=args.ch, res_num=args.res_num, C_Q=args.C_Q)
            )
        self.encoders.append(EncUnitLast(in_ch=args.ch, out_ch=args.ch, res_num=args.res_num, C_Q=args.C_Q))
        self.decoders = nn.ModuleList()
        #decoder 0->s-1 corespoding to s-1->0
        self.decoders.append(DecUnitFirst(in_ch=args.ch, C_Q=args.C_Q, res_num=args.res_num, param_num=args.K*args.C_Q*3))
        for i in range(self.layers-2):
            self.decoders.append(
                DecUnit(in_ch=args.ch, C_Q=args.C_Q, res_num=args.res_num, param_num=args.K*args.C_Q*3)
            )
        self.decoders.append(
            DecUnit(in_ch=args.ch, C_Q=args.C_Q, res_num=args.res_num, param_num=12*args.K)
        )
        self.quantizer = Quantizer()
        self.logistic_mixture = LogisticMixture()
        self.uniform = UniformDistribution()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x:Tensor):
        """
        x:[B,C,H,W] rgb img
        """
        z_hat_list = []
        #encoder tower
        x_input = self.meanshift(x)
        x_input = self.head(x_input)
        input_encoder, out_q1 = self.encoders[0](x_input)
        z_hat = self.quantizer.quantize(out_q1, min=self.z_min, max=self.z_max, level=self.z_level, mode="soft",softness=self.softness)
        z_hat_list.append(z_hat)
        neg_log_likelihood = []
        
        for i in range(self.layers-2):
            input_encoder, out_qi = self.encoders[i+1](input_encoder)
            z_hat_list.append(self.quantizer.quantize(
                out_qi, min=self.z_min, max=self.z_max, 
                level=self.z_level, 
                mode="soft", softness=self.softness
            ))
        z_s = self.encoders[self.layers-1](input_encoder)
        z_hat_list.append(
            self.quantizer.quantize(
                z_s, min=self.z_min,
                max=self.z_max, level=self.z_level,
                mode="soft", softness=self.softness
            )
        )


#self, quantized_x:Tensor, params:Tensor, param_num:int, bin:float, x_min:float, x_max:float
#默认layers>1
        #for z_s
        neg_log_likelihood.append(
            self.uniform(z_hat_list[-1], self.z_level)
        )

        for i in range(self.layers):
            if i==0:  
                F_s, params = self.decoders[i](z_hat_list[self.layers-1-i])
                neg_log_likelihood.append(
                    self.logistic_mixture(
                        quantized_x=z_hat_list[self.layers-2-i],
                        params=params,
                        param_num=3,
                        bin=self.z_bin,
                        x_min=self.z_min,
                        x_max=self.z_max,
                    )
                )
            elif i==self.layers-1:
                _, params = self.decoders[i](z_hat_list[self.layers-1-i], F_s)
                neg_log_likelihood.append(
                    self.logistic_mixture(
                        quantized_x=x,
                        params=params,
                        param_num=4,
                        bin=1,
                        x_min=0,
                        x_max=255
                    )
                )
            else:
                F_s, params = self.decoders[i](z_hat_list[self.layers-1-i], F_s)
                neg_log_likelihood.append(
                    self.logistic_mixture(
                        quantized_x=z_hat_list[self.layers-2-i],
                        params=params,
                        param_num=3,
                        bin=self.z_bin,
                        x_min=self.z_min,
                        x_max=self.z_max
                    )
                )
        return neg_log_likelihood
    

    def compress(self, x:Tensor):
        """
        x:tensor[B,C,H,W]
        return z_symbol_list, cdf_list
        cannot return string_list, because torchac must run on cpu
        """
        z_hat_list = []
        #encoder tower
        x_input = self.meanshift(x)
        x_input = self.head(x_input)
        input_encoder, out_q1 = self.encoders[0](x_input)
        z_hat_list.append(self.quantizer.quantize(out_q1, min=self.z_min, 
                                                  max=self.z_max, level=self.z_level,
                                                  mode="hard"))
        for i in range(self.layers-2):
            input_encoder, out_qi = self.encoders[i+1](input_encoder)
            z_hat_list.append(self.quantizer.quantize(
                out_qi, min=self.z_min, max=self.z_max,
                level=self.z_level,
                mode="hard"
            ))
        z_s = self.encoders[self.layers-1](input_encoder)
        z_hat_list.append(
            self.quantizer.quantize(
                z_s, min=self.z_min,
                max=self.z_max, level=self.z_level,
                mode="hard"
            )
        )       
        string_list = []
        string_rgb = []
        #encoding z_s
        cdf_uniform = self.uniform.get_cdf(z_hat_list[-1].shape, device=z_hat_list[-1].device, level=self.z_level)
        z_sym = self.quantizer.quantize(z_hat_list[-1], min=self.z_min, max=self.z_max, level=self.z_level, mode="symbol").to(torch.int16)
        string_list.append(bitencode(sym=z_sym, cdf=cdf_uniform))
        for i in range(self.layers):
            if i==0:  
                F_s, params = self.decoders[i](z_hat_list[self.layers-1-i])
                miu, log_scale, weight = self.logistic_mixture.param_prepare_cdf_z(z_hat_list[self.layers-2-i].shape, params, 3)
                cdf_logistic = self.logistic_mixture.get_cdf(
                    shape=z_hat_list[self.layers-2-i].shape,
                    miu=miu,
                    log_scale=log_scale,
                    weight=weight,
                    level_num=self.z_level,
                    bin = self.z_bin,
                    x_min = self.z_min,
                    x_max = self.z_max
                )
                z_sym=self.quantizer.quantize(
                    z_hat_list[self.layers-2-i],
                    min=self.z_min,
                    max=self.z_max,
                    level=self.z_level,
                    mode="symbol").to(torch.int16)
                #encoding z(s-1)
                string_list.append(bitencode(sym=z_sym, cdf=cdf_logistic))
                del z_sym
                del cdf_logistic
                
            elif i==self.layers-1:
                #encoding x, need to encode r,g,b separately
                _, params = self.decoders[i](z_hat_list[self.layers-1-i], F_s)
                #cdf logistic: [B,3,H,W,257]
                decoded_list = []
                for j in range(3):
                    miu, log_scale, weight = self.logistic_mixture.param_prepare_cdf_x(x.shape, j, decoded=decoded_list, params=params)
                    cdf_logistic = self.logistic_mixture.get_cdf(
                        shape=x[:,j,:,:].unsqueeze(1).shape,
                        miu=miu,
                        log_scale=log_scale,
                        weight=weight,
                        level_num=256,
                        bin=1,
                        x_min=0,
                        x_max=255
                    )
                    string_rgb.append(bitencode(sym=x[:,j,:,:].unsqueeze(1).to(torch.int16), cdf=cdf_logistic))
                    decoded_list.append(x[:,j,:,:])
                    del cdf_logistic
                
            else:
                F_s, params = self.decoders[i](z_hat_list[self.layers-1-i],F_s)
                miu, log_scale, weight = self.logistic_mixture.param_prepare_cdf_z(z_hat_list[self.layers-2-i].shape, params, 3)
                cdf_logistic = self.logistic_mixture.get_cdf(
                    shape=z_hat_list[self.layers-2-i].shape,
                    miu=miu,
                    log_scale=log_scale,
                    weight=weight,
                    level_num=self.z_level,
                    bin = self.z_bin,
                    x_min = self.z_min,
                    x_max = self.z_max
                )
                z_sym=self.quantizer.quantize(
                    z_hat_list[self.layers-2-i],
                    min=self.z_min,
                    max=self.z_max,
                    level=self.z_level,
                    mode="symbol").to(torch.int16)
                #encoding z(s-1)
                string_list.append(bitencode(sym=z_sym, cdf=cdf_logistic))
                del z_sym
                del cdf_logistic
        return {
            "z_strings":string_list,
            "x_strings":string_rgb,
            "z_shape":z_hat_list[-1].shape,
            "z_hat_list":z_hat_list
        }


    def decompress(self, string_z, string_rgb, z_s_shape):
        #decode z(s) using uniform distribution
        device = next(self.parameters()).device
        cdf_uniform = self.uniform.get_cdf(shape=z_s_shape, device=device, level=self.z_level)
        z_input_index = bitdecode(cdf=cdf_uniform, string=string_z[0]).to(torch.long).to(device)
        z_input = self.quantizer.dequantize(z_input_index, min=self.z_min, max=self.z_max, level=self.z_level)
        for i in range(self.layers):
            if i==0:
                F_s, params = self.decoders[i](z_input)
                B,C,H,W = params.shape
                miu, log_scale, weight = self.logistic_mixture.param_prepare_cdf_z(
                    shape=(B,C//self.K//3,H,W),
                    params=params,
                    param_num=3
                )
                cdf_logistic = self.logistic_mixture.get_cdf(
                    shape=(B,C//self.K//3,H,W),
                    miu=miu,
                    log_scale=log_scale,
                    weight=weight,
                    level_num=self.z_level,
                    bin=self.z_bin,
                    x_min=self.z_min,
                    x_max=self.z_max
                )
                z_input_index = bitdecode(cdf=cdf_logistic, string=string_z[i+1]).to(torch.long).to(device)
                z_input = self.quantizer.dequantize(z_input_index, min=self.z_min, max=self.z_max, level=self.z_level)
            elif i==self.layers-1:
                _, params = self.decoders[i](z_input, F_s)
                B,C,H,W = params.shape
                decoded_list = []
                for j in range(3):
                    miu, log_scale, weight = self.logistic_mixture.param_prepare_cdf_x(
                        shape=(B,3,H,W),
                        channel=j,
                        decoded=decoded_list,
                        params=params,
                    )
                    cdf_logistic = self.logistic_mixture.get_cdf(
                        shape=(B,1,H,W),
                        miu=miu,
                        log_scale=log_scale,
                        weight=weight,
                        level_num=256,
                        bin=1,
                        x_min=0,
                        x_max=255
                    )
                    decoded_list.append(bitdecode(cdf_logistic, string=string_rgb[j]).to(torch.float32).to(device))
            else:
                F_s, params = self.decoders[i](z_input, F_s)
                B,C,H,W = params.shape
                miu, log_scale, weight = self.logistic_mixture.param_prepare_cdf_z(
                    shape=(B,C//self.K//3, H,W),
                    params=params,
                    param_num = 3,
                )
                cdf_logistic = self.logistic_mixture.get_cdf(
                    shape=(B,C//self.K//3,H,W),
                    miu=miu,
                    log_scale=log_scale,
                    weight=weight,
                    level_num=self.z_level,
                    bin=self.z_bin,
                    x_min=self.z_min,
                    x_max=self.z_max
                )
                z_input_index = bitdecode(cdf=cdf_logistic, string=string_z[i+1]).to(torch.long).to(device)
                z_input = self.quantizer.dequantize(index=z_input_index, min=self.z_min, max=self.z_max, level=self.z_level)
        
        return torch.cat(decoded_list,dim=1)
    
    
    def get_latent(self, x):
        z_hat_list = []
        #encoder tower
        x_input = self.meanshift(x)
        x_input = self.head(x_input)
        input_encoder, out_q1 = self.encoders[0](x_input)
        z_hat_list.append(self.quantizer.quantize(out_q1, min=self.z_min, 
                                                  max=self.z_max, level=self.z_level,
                                                  mode="hard"))
        for i in range(self.layers-2):
            input_encoder, out_qi = self.encoders[i+1](input_encoder)
            z_hat_list.append(self.quantizer.quantize(
                out_qi, min=self.z_min, max=self.z_max,
                level=self.z_level,
                mode="hard"
            ))
        z_s = self.encoders[self.layers-1](input_encoder)
        z_hat_list.append(
            self.quantizer.quantize(
                z_s, min=self.z_min,
                max=self.z_max, level=self.z_level,
                mode="hard"
            )
        )       
        return z_hat_list

if __name__=="__main__":
    random.seed(6)
    torch.manual_seed(6)
    torch.cuda.manual_seed_all(6)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=False
    net_config = NetConfig()
    net = Network(net_config).to("cuda")
    x = torch.round(torch.rand([1,3,1000,1000])*255).to("cuda")
    print(net)
    #out_net =  net(x)
    with torch.no_grad():
        out_enc = net.compress(x)
        out_dec = net.decompress(string_z=out_enc["z_strings"],string_rgb=out_enc["x_strings"],z_s_shape=out_enc["z_shape"])
    assert torch.equal(out_dec,x)
    """
    net_input = (3, 128, 128)
    macs, params = get_model_complexity_info(net, net_input, print_per_layer_stat=True, as_strings=True)
    print("macs:{}   params:{}".format(macs, params))
    """