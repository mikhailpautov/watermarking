import torch
import torchvision
import argparse

from datasets import get_dataset, DATASETS, get_normalize_layer
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from global_options import _models, _optimizers
from aux import train, accuracy




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'mnist', 'cifar100'], default='cifar10')
    parser.add_argument('--epochs', help="n epochs", type=int, default=50)
    parser.add_argument('--batch', help="batch_size", type=int, default=32)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=5)
    parser.add_argument('--results_dir', help="dir for results", type=str, default='./results')
    parser.add_argument('--lr', help="learning rate", type=float, default=1e-3)
    parser.add_argument('--momentum', help="momentum factor", type=float, default=.95)
    parser.add_argument('--val_freq', help="how often eval model during training", type=int, default=1)
    
    parser.add_argument('--model_name', help = "model architecture", choices=list(_models.keys()), required=True)
    parser.add_argument('--optimizer', help= "optimizer to train a model", choices=list(_optimizers.keys()), required=True)
    parser.add_argument('--workers', help="n workers", type=int, default=8)
    parser.add_argument('--pin_memory', help="pin mem?", type=bool, default=False)
    parser.add_argument('--outdir', help="direct where to save sht", type=str, default='./')
    parser.add_argument('--print_freq', help="how hard it is 4 me to shake the disease", type=int, default=10)
    parser.add_argument('--print_step', type=int, default=10)
    parser.add_argument('--run_name', type=int, required=True)
    
    args = parser.parse_args()


    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                            num_workers = args.workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                            num_workers = args.workers, pin_memory=args.pin_memory)
    
    
    model = _models[args.model_name]()
    optimizer = _optimizers[args.optimizer](model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = CrossEntropyLoss().to(args.device)
    
    args.device = torch.device(args.device)
    expname = '_' + args.dataset + '_' + str(args.run_name)
    args.expname = expname
    
    
    
    train(model=model, optimizer=optimizer, train_dataloader=train_loader, test_dataloader=test_loader, criterion=criterion, args=args)