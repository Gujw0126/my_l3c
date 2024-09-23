import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import torch.nn as nn
from PIL import Image 
import csv
import sys
import time
from Network import*
import argparse
from torchvision import transforms as transforms
import math

EXTENSIONS = [
    '.jpg',
    '.png'
]


def collect_images(root):
    return[os.path.join(root, filename) for filename in os.listdir(root)
           if os.path.splitext(filename)[-1].lower() in EXTENSIONS]


def read_img(img_path):
    trans = transforms.Compose([transforms.PILToTensor()])
    img = trans(Image.open(img_path).convert('RGB')).to(torch.float32)
    return img



def test_all(file_paths, net):
    net.eval()
    device = next(net.parameters()).device
    total_compress_time = 0
    total_decompress_time = 0
    total_bpsp = 0

    with torch.no_grad():
        for img_path in file_paths:
            x = read_img(img_path).to("cuda").unsqueeze(0)
            compress_start_time = time.time()
            B,C,H,W = x.shape
            subpixel_num = B*C*H*W
            new_h = math.ceil(H/8)*8
            new_w = math.ceil(W/8)*8
            padding_left = (new_w-W)//2
            padding_right = new_w - W - padding_left
            padding_top = (new_h - H)//2
            padding_down = new_h - H - padding_top
            x_padded = nn.functional.pad(x, [padding_left, padding_right, padding_top, padding_down], mode="constant", value=0)
            out_compress = net.compress(x_padded)
            compress_use_time = time.time() - compress_start_time
            z3_bpsp = len(out_compress["z_strings"][0])/subpixel_num*8
            z2_bpsp = len(out_compress["z_strings"][1])/subpixel_num*8
            z1_bpsp = len(out_compress["z_strings"][2])/subpixel_num*8
            x_bpsp = (len(out_compress["x_strings"][0])+len(out_compress["x_strings"][1])+len(out_compress["x_strings"][2]))/subpixel_num*8
            bpsp = z3_bpsp + z2_bpsp + z1_bpsp + x_bpsp    
            #decompress
            decompress_start_time = time.time()
            out_decompress = net.decompress(out_compress["z_strings"], out_compress["x_strings"], out_compress["z_shape"])
            x_hat = nn.functional.pad(out_decompress,[-padding_left, -padding_right ,-padding_top, -padding_down])
            decompress_use_time = time.time() - decompress_start_time
            assert(x_hat.equal(x))

            #write csv_file and statistic
            total_bpsp += bpsp
            img_name = os.path.splitext(img_path.split('/')[-1])[-2]
            print("img_name{}\t|bpsp:{:.4f}\t|z3_bpsp:{:.4f}\t|z2_bpsp:{:.4f}\t|z0_bpsp:{:.4f} \
              \t|x_bpsp:{:.4f}\t|compress_time:{:.4f}\t|decompress_time:{:.4f}".format(img_name,bpsp,z3_bpsp,z2_bpsp,z1_bpsp,x_bpsp,compress_use_time, decompress_use_time))
    
    print("avg_bpsp:{:.4f}".format(total_bpsp/len(file_paths)))


def inference_all(file_paths, net):
    net.eval()
    total_bpsp = 0

    with torch.no_grad():
        for img_path in file_paths:
            x = read_img(img_path).to("cuda:0").unsqueeze(0)
            B,C,H,W = x.shape
            subpixel_num = B*C*H*W
            new_h = math.ceil(H/8)*8
            new_w = math.ceil(W/8)*8
            padding_left = (new_w-W)//2
            padding_right = new_w - W - padding_left
            padding_top = (new_h - H)//2
            padding_down = new_h - H - padding_top
            x_padded = nn.functional.pad(x, [padding_left, padding_right, padding_top, padding_down], mode="constant", value=0)
            out_net = net(x_padded)
            x_bpsp = 0
            z3_bpsp = 0
            z2_bpsp = 0
            z1_bpsp = 0
            z3_bpsp += out_net[0].sum()/subpixel_num
            z2_bpsp += out_net[1].sum()/subpixel_num
            z1_bpsp += out_net[2].sum()/subpixel_num
            x_bpsp += out_net[3].sum()/subpixel_num
            bpsp = x_bpsp + z1_bpsp+z2_bpsp+z3_bpsp
            #write csv_file and statistic
            total_bpsp += bpsp
            img_name = os.path.splitext(img_path.split('/')[-1])[-2]
            print("img_name{}\t|bpsp:{:.4f}\t|:z3_bpsp{:.4f}\t|z2_bpsp{:.4f}\t|z1_bpsp{:.4f}".format(img_name,bpsp,z3_bpsp,z2_bpsp,z1_bpsp))
    
    print("avg_bpsp:{:.4f}".format(total_bpsp/len(file_paths)))


def set_args(argv):
    parser = argparse.ArgumentParser(description="test a model from checkpoint")
    parser.add_argument("-d","--dataset", required=True, type=str, help="dataset to test")
    parser.add_argument("-c","--checkpoint", required=True, type=str, help="checkpoint path")
    parser.add_argument("--seed",default=1926, type=int, help="random seed")
    parser.add_argument("--savepath", required=True, type=str, help="path to save csv file")
    parser.add_argument("--cuda", action="store_true", help="use gpu or not")
    parser.add_argument("--gpu-id", type=str, default="0",help="gpu id")
    return parser.parse_args(argv)


def main(argv):
    args = set_args(argv)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    config = NetConfig()
    model = Network(config)
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    #print(model.state_dict().keys())
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if not key.startswith('module.'):
            new_key = 'module.' + key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)    
    """
    model.load_state_dict(state_dict)
    model.eval().to("cuda:0")
    file_paths = collect_images(args.dataset)
    #test_all(file_paths, model, args.savepath)
    test_all(file_paths, model)



if __name__=="__main__":
    main(sys.argv[1:])
