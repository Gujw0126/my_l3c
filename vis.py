#visualize our model's latent
import torch
import torch.nn as nn
from matplotlib import pyplot
import seaborn as sns
from Network import Network,NetConfig
import os
from PIL import Image
from torchvision import transforms

def vis_latent(z_hat_list,savepath):
    for layer in range(len(z_hat_list)):
        for ch in range(5):
            fig = pyplot.figure()
            sns.heatmap(z_hat_list[layer][0,ch,:,:].to("cpu"))
            img_name = os.path.join(savepath, "z_hat{}_channel{}.png".format(layer,ch))
            pyplot.savefig(img_name)
            pyplot.close(fig)


def main():
    net_config = NetConfig()
    model = Network(net_config).eval().to("cuda")
    img_path = "your image path"
    checkpoint_path = "your checkpoint"
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict["model_state_dict"])
    img = Image.open(img_path).convert("RGB")
    transform = transforms.PILToTensor()
    img = transform(img)
    z_hat_list = model.get_latent(img.to(torch.float).to("cuda").unsqueeze(0))
    vis_latent(z_hat_list,"vis_result")


if __name__=="__main__":
    main()