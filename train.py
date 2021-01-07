import os
import sys
import datetime
import time

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from cifar10 import CIFAR10
from mobilenetv2 import DyMobileNetV2
from utils import select_device, increment_path, Logger, AverageMeter, save_model, \
    print_argument_options, init_torch_seeds


def main(opt, device):

    if not opt.nlog:
        sys.stdout = Logger(Path(opt.save_dir) / 'log_.txt')
    print_argument_options(opt)
    
    #Configure
    cuda = device.type != 'cpu'
    init_torch_seeds()

    dataset = CIFAR10(opt.batch_size, cuda, opt.workers)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    opt.num_classes = dataset.num_classes
    print("Creat dataset: {}".format(dataset.__class__.__name__))

    model = DyMobileNetV2(num_classes=opt.num_classes, input_size=32, width_mult=1.).to(device)
    
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("Creat model: {}".format(model.__class__.__name__))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, weight_decay=5e-04, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
    
    opt.scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_time = time.time()    
    for epoch in range(opt.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, opt.max_epoch))

        if opt.training_optim: # It only faster on GPU
            model.training_mode()
        else:
            model.inference_mode()

        __training(opt, model, criterion, optimizer, trainloader, epoch, device)
        scheduler.step()

        if opt.eval_freq > 0 and (epoch+1) % opt.eval_freq == 0 or (epoch+1) == opt.max_epoch:
            acc, err = __testing(opt, model, trainloader, epoch, device)
            print("==> Train Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            acc, err = __testing(opt, model, testloader, epoch, device)
            print("==> Test Accuracy (%): {}\t Error rate(%): {}".format(acc, err))
            save_model(model, epoch, name=opt.model, save_dir=opt.save_dir)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def __training(opt, model, criterion, optimizer, trainloader, epoch, device):
    model.train()
    losses = AverageMeter()
    
    start_time = time.time() 
    for i, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, labels)
        opt.scaler.scale(loss).backward()
        opt.scaler.step(optimizer)
        opt.scaler.update()

        optimizer.zero_grad()
        losses.update(loss.item(), labels.size(0))
                 
        if (i+1) % opt.print_freq == 0:
            elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            start_time = time.time()
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) elapsed time (h:m:s): {}" \
                .format(i+1, len(trainloader), losses.val, losses.avg, elapsed))
            

def __testing(opt, model, testloader, epoch, device):
    model.eval()
    correct, total = 0, 0
                
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err
    

def parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr'               , default=0.1)
    parser.add_argument('--workers'          , default=4)
    parser.add_argument('--batch_size'       , default=256)
    parser.add_argument('--max_epoch'        , default=100)
    parser.add_argument('--stepsize'         , default=30)
    parser.add_argument('--gamma'            , default=0.1)
    parser.add_argument('--training_optim', action='store_true', help='training more faster')

    parser.add_argument('--eval_freq'        , default=10)
    parser.add_argument('--print_freq'       , default=50)
    parser.add_argument('--nlog', action='store_true', help='nlog = not print log.txt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    return parser.parse_args()

if __name__ == "__main__":
    opt = parser()
    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.save_dir = increment_path(Path(opt.project) / 'cifar10' / 'mobilenetv2' / opt.name, exist_ok=opt.exist_ok)  # increment run
    
    main(opt, device)

    
