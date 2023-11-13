import torch
import os
import sys

from datasets import get_dataset, DATASETS, get_normalize_layer
from time import time
from tqdm import tqdm
from torch.optim import SGD, Adam



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()





def train(model, train_dataloader, test_dataloader, criterion, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model_device = next(model.parameters()).device

    if model_device != args.device:
        model = model.to(args.device)
        
    model.train()
    optimizer = _optimizers[args.optimizer](model.parameters(), lr=args.lr, momentum=args.momentum)
    
    n_epochs = args.epochs
    
    for epoch_num in range(n_epochs):
            
        for batch in tqdm(train_dataloader):
                
            inputs, targets = batch
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
        
            b_size = inputs.size(0)
            
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            loss = criterion(outputs, targets)
            losses.update(loss.item(), b_size)

        
            top1.update(acc1.item(), b_size)
            top5.update(acc5.item(), b_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    if epoch_num % args.print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch_num, loss=losses, top1=top1, top5=top5))
            
        torch.save({
        'epoch': epoch_num + 1,
        'model_name': args.model_name,
        'state_dict': model.state_dict(),
        }, os.path.join(args.outdir, args.expname+ '_checkpoint.pth.tar'))        