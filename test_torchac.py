import torchac
import torch
import os
from PIL import Image
from torchvision import transforms



def test():
    cdf=torch.tensor([0,0.25,0.5,0.75,1])
    cdf = cdf.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    cdf = cdf.expand([1,1,2,2,5])
    sym = torch.tensor([[0,1],[2,3]], dtype=torch.int16)
    sym = sym.unsqueeze(0).unsqueeze(0)
    bytestream  = torchac.encode_float_cdf(cdf_float=cdf, sym=sym, check_input_bounds=True)
    length = len(bytestream)*8
    print("length:{}".format(length))
    decoded = torchac.decode_float_cdf(cdf, bytestream)
    assert decoded.equal(sym)


def find_biggest_img(dataset):
    max_pixel = 0
    max_H = 0
    max_W = 0
    for imgpath in os.listdir(dataset):
        img = Image.open(os.path.join(dataset, imgpath)).convert("RGB")
        img_torch = transforms.ToTensor()(img)
        C,H,W = img_torch.shape
        pixel_num = H*W
        if pixel_num>max_pixel:
            max_pixel = pixel_num
            max_H = H
            max_W = W
    print("max_pixel_num:{}".format(max_pixel))
    print("max_H:{}".format(max_H))
    print("max_W:{}".format(max_W))


if __name__=="__main__":
    #test()
    find_biggest_img("/mnt/data3/jingwengu/dataset/DIV2K_valid_HR")
