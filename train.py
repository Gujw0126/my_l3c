import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch
import torch.nn as nn
import sys
import argparse
from torch.utils.data import DataLoader
from typing import List
import time
import random
from Network import Network, NetConfig
import torch.optim as optim
from scheduler import ExponentialDecayLRSchedule
from L3C import ImageFolder
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import logging
my_logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, Tuple


def DelfileList(path, filestarts='checkpoint_last'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(filestarts):
                os.remove(os.path.join(root, file))


class Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, neg_likelihood:List):
        B,C,H,W = neg_likelihood[-1].shape
        subpixel_num = B*C*H*W
        loss = 0
        z_loss  = []
        layers = len(neg_likelihood)
        for i in range(layers):
            if i>0:
                loss += torch.sum(neg_likelihood[i])/subpixel_num
            z_loss.append(torch.sum(neg_likelihood[i])/subpixel_num)
        return {
            "total_loss":loss,
            "z_loss":z_loss
        }


class AverageMeter():
    def __init__(self) -> None:
        self.value=0
        self.sum=0
        self.count=0
        self.avg=0
    
    def update(self, value, n=1):
        self.count += n
        self.value = value
        self.sum += value*n
        self.avg = self.sum/self.count


def save_checkpoint(savepath, name, iteration:int, epoch:int, model_state, optimizer_state_dict, scheduler_state_dict):
    checkpoint_name = "checkpoint_"+name+"_{}.pth.tar".format(iteration)
    checkpoint_path = os.path.join(savepath, checkpoint_name)
    checkpoint = {"model_state_dict":model_state,
                  "optimizer_state_dict":optimizer_state_dict,
                  "scheduler_state_dict":scheduler_state_dict,
                  "iteration":iteration,
                  "epoch":epoch}
    torch.save(checkpoint, checkpoint_path)


#train_one_epoch, but save every 1000 iteration
def train_one_epoch(net, optimizer, lr_scheduler, epoch, iteration_before, criterion, dataloader, logger, summery_writer, savepath_train):
    torch.autograd.set_detect_anomaly(True)
    device = next(net.parameters()).device
    train_loss = AverageMeter()
    iteration_after = iteration_before
    z_loss_avg_meters = []
    for i in range(net.layers+1):
        z_loss_avg_meters.append(AverageMeter())

    start_time = time.time()
    if epoch % 5==0 and epoch>0:
        lr_scheduler.step()
    print("---------------------------epoch{}------------------------------------".format(epoch))
    logger.info("---------------------------epoch{}------------------------------------".format(epoch))
    #start training this epoch
    for i, x in enumerate(dataloader):
        x = x.to(device).to(torch.float32)
        optimizer.zero_grad()
        out_net = net(x)
        out_criterion = criterion(out_net)
        train_loss.update(out_criterion["total_loss"].item())
        for j in range(net.layers+1):
            z_loss_avg_meters[j].update(out_criterion["z_loss"][j].item())
        out_criterion["total_loss"].backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        optimizer.step()
        iteration_after += 1

        #print, log and write summery every 100 iteration
        if iteration_after % 100 == 0:
            #print and log
            z_loss_info = ""
            for scale in range(net.layers+1):
                z_loss_info += "|z({}):{:4f}\t".format(net.layers-scale, z_loss_avg_meters[scale].avg)
            info_first = "Train: iteration:{}\t|learning_rate:{}\t|epoch:{}\t|loss:{:.4f}\t".format(iteration_after, optimizer.param_groups[0]['lr'],epoch, train_loss.avg)
            info_last = "|time:{:.4f}".format(time.time()-start_time)
            start_time = time.time()
            info = info_first + z_loss_info
            print(info)
            logger.info(info)
            #write summery
            summery_writer.add_scalar('Train/total_loss(bpsp)', train_loss.avg, iteration_after)
            for scale in range(net.layers+1):
                summery_writer.add_scalar("Train/z{}_bpsp".format(net.layers-scale), z_loss_avg_meters[scale].avg, iteration_after)
        
        #save checkpoint every 1000 iteration
        if iteration_after %1000 == 0:
            print("saving checkpoint...")
            save_checkpoint(
                savepath=savepath_train,
                iteration=iteration_after,
                epoch=epoch,
                model_state=net.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=lr_scheduler.state_dict(),
                name="train"
            )

    z_loss_info = ""
    for j in range(net.layers+1):
        z_loss_info += "|z({}):{:4f}\t".format(net.layers-j, z_loss_avg_meters[j].avg)
    info_first = "one epoch over! Train: iteration:{}\t|epoch:{}\t|loss:{:.4f}\t".format(iteration_after, epoch, train_loss.avg)
    info_last = "|time:{:.4f}".format(time.time()-start_time)
    info = info_first + z_loss_info + info_last
    print(info)
    logger.info(info)
    return train_loss.avg, [z_loss_avg_meters[j].avg for j in range(net.layers+1)], iteration_after




@torch.no_grad()
def test_one_epoch(net, epoch, criterion, dataloader,logger):
    device = next(net.parameters()).device
    test_loss = AverageMeter()
    z_loss_avg_meters = []
    for i in range(net.layers+1):
        z_loss_avg_meters.append(AverageMeter())

    start_time = time.time()
    for i, x in enumerate(dataloader):
        x = x.to(device).to(torch.float32)
        out_net = net(x)
        out_criterion = criterion(out_net)
        test_loss.update(out_criterion["total_loss"].item())
        for j in range(net.layers+1):
            z_loss_avg_meters[j].update(out_criterion["z_loss"][j].item())

    z_loss_info = ""
    for j in range(net.layers+1):
        z_loss_info += "|z({}):{:4f}\t".format(net.layers-j, z_loss_avg_meters[j].avg)
    info_first = "Test: epoch:{}\t|loss:{:.4f}\t".format(epoch, test_loss.avg)
    info_last = "|time:{:.4f}".format(time.time()-start_time)
    info = info_first + z_loss_info + info_last
    print(info)
    logger.info(info)
    return test_loss.avg, [z_loss_avg_meters[j].avg for j in range(net.layers+1)]


def set_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, required=True, help="path to dataset")
    parser.add_argument("-e","--epoch", type=int, default=1000, help="training epoch")
    parser.add_argument("-n","--numworkers", type=int, default=4, help="num_workers")
    parser.add_argument("-lr","--learning_rate", type=float, default=1e-4,help="learning_rate")
    parser.add_argument("--decay",type=float, default=0, help="weight decay")
    parser.add_argument("--decay-e", type=int, default=5, help="learning rate decay every epoch")
    parser.add_argument("-f","--fac", type=float, default=0.75, help="learning rate decay factor")
    parser.add_argument("--batch-size", type=int, default=30, help="training batch-size")
    parser.add_argument("--test-batch", type=int, default=30, help="test batch-size")
    parser.add_argument("--patch-size",type=int ,nargs=2,default=(128,128),help="crop_size")
    #parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--seed",type=int, default=1926, help="random seed used for reproductivity")
    parser.add_argument("--cuda", action="store_true", help="use cuda or not")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    parser.add_argument("--savepath", type=str, required=True, help="path to save checkpoint")
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")
    args  = parser.parse_args(argv)
    return args


def main(argv):
    args = set_args(argv)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    log_file = os.path.join(args.savepath,"train.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    savepath_itration = os.path.join(args.savepath, "iteration")
    savepath_epoch = os.path.join(args.savepath, "epoch")
    if not os.path.exists(savepath_itration):
        os.mkdir(savepath_itration)
    if not os.path.exists(savepath_epoch):
        os.mkdir(savepath_epoch)

    if (not os.path.isdir(savepath_epoch)) or (not os.path.isdir(savepath_itration)):
        raise RuntimeError("savepath error")
    

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


    net_config = NetConfig()
    net = Network(net_config).to(device)
    #net.initialize()
    optimizer = optim.RMSprop(net.parameters(),lr=args.learning_rate, weight_decay=args.decay)
    last_epoch = 0
    last_iteration = 0
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(size=args.patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()]
    )
    
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(size=args.patch_size),
        transforms.PILToTensor()]
    )

    train_dataset = ImageFolder(root=args.dataset, split="train",transform=train_transforms)
    test_dataset = ImageFolder(root=args.dataset, split="val", transform=test_transforms)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.numworkers,
        pin_memory=True,
        pin_memory_device=device
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.numworkers,
        pin_memory=True,
        pin_memory_device=device
    )
    """
    lr_schedule = ExponentialDecayLRSchedule(
        optims=optimizer,
        initial=args.lr,
        decay_fac=args.fac,
        decay_interval_itr=None,
        decay_interval_epoch=args.decay_e,
        epoch_len=len(train_dataloader)
        )
    """
    writer = SummaryWriter(log_dir=args)
    expr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.fac)

    #from checkpoint load state_dict, optimizer_state_dict and lr_schedule state
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        expr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        last_epoch = checkpoint["epoch"]+1
        last_iteration = checkpoint["iteration"]+1
    
    #training
    criterion = Loss()
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epoch):
        #add function save every 1000 iteration
        train_loss, z_loss_list, last_iteration = train_one_epoch(net, 
                                     optimizer=optimizer, 
                                     lr_scheduler=expr_scheduler, 
                                     epoch=epoch,criterion=criterion,
                                     iteration_before=last_iteration,
                                     logger=my_logger,
                                     dataloader=train_dataloader,
                                     summery_writer=writer,
                                     savepath_train=savepath_itration
                                     )
        
        writer.add_scalar('Train_epoch/loss', train_loss, epoch)
        for scale in range(net.layers+1):
            writer.add_scalar('Train_epoch/z({})_loss'.format(net.layers-scale),z_loss_list[scale],epoch)
        test_loss, z_loss_test = test_one_epoch(net=net, 
                                   epoch=epoch, 
                                   criterion=criterion,
                                   dataloader=test_dataloader,
                                   logger=my_logger
                                   )
        writer.add_scalar('Test/loss', test_loss, epoch)
        for scale in range(net.layers+1):
            writer.add_scalar('Test/z({})_loss'.format(net.layers-scale),z_loss_test[scale],epoch)

        save_checkpoint(
            savepath=args.savepath,
            epoch=epoch,
            model_state=net.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=expr_scheduler.state_dict(),
            name="last",
            iteration=last_iteration
        )
        if test_loss<best_loss:
            best_loss = test_loss
            save_checkpoint(
                savepath=args.savepath,
                epoch=epoch,
                model_state=net.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=expr_scheduler.state_dict(),
                name="best",
                iteration=last_iteration
            )


if __name__=="__main__":
    debug=["-d", "/mnt/data3/jingwengu/dataset/openimage", "-e", "100" ,"--cuda", "--savepath", "/mnt/data3/jingwengu/my_l3c"]
    main(sys.argv[1:])