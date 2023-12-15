import torch
import torchvision
import argparse
import copy

import torchvision.transforms as transforms

from datasets import get_dataset, get_bounds, get_num_classes
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from global_options import _models, _optimizers
from aux import train, accuracy, flatten_params, recover_flattened, evaluate_model

import foolbox as fb
import matplotlib.pyplot as plt

import glob

from models import resnet18, resnet34, resnet50



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'mnist'], default='cifar10')
    parser.add_argument('--batch', help="batch_size", type=int, default=32)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=5)
    parser.add_argument('--model_name', help = "target model architecture", choices=list(_models.keys()), default='resnet50')
    parser.add_argument('--model_path', help="path to saved model", type=str, default='./model_0.pt')
    parser.add_argument('--N', help="size of trigger set", type=int, default=32)
    
    args = parser.parse_args()

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    num_classes = get_num_classes(args.dataset)
    bounds = get_bounds(args.dataset)
    
    # model = _models[args.model_name](num_classes=num_classes)
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model = resnet34()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu')['model'])
    model.eval()
    
    
    ### Collect adversarial examples ###
    
    fmodel = fb.PyTorchModel(model, bounds=bounds, device=args.device)
    attack = fb.attacks.LinfFastGradientAttack(random_start=True)
    
    epsilons = 0.02  # for adversaarial examples
    I = 0
    adv_dataset = []
    
    for image, label in test_dataset: # train_dataset or test_dataset ?
        logits = model(image.unsqueeze(dim=0).to(args.device))
        prediction = torch.argmax(logits, dim=-1)
    
        if not label == prediction.item():
            print("Wrong answer!")
            continue
        
        images = image.repeat(args.batch, 1, 1, 1)
        labels = label * torch.ones(args.batch)
        labels = labels.long()
        images, labels = images.to(args.device), labels.to(args.device)
         
        _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        
        if not all(success):
            print("Fail attack")
            continue
        
        logits = model(advs)
        target_predictions = torch.argmax(logits, dim=-1)
        
        adv_dataset.append((advs.detach().cpu(), target_predictions.detach().cpu()))
        I += 1

        if I == args.N:
            break
        
        
    ### Reject sampling. Compute predictions of proxyb models ###
    
    model.to('cpu')
    flatten_dict = flatten_params(model.parameters())
    init_params, init_indices = flatten_dict['params'], flatten_dict['indices']    

    FLAGS = []
    for i in range(100):
        model_copy = copy.deepcopy(model)
    
        sigma = 1e-3    
        delta = sigma * torch.randn_like(init_params)
        new_params = init_params + delta
        new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
        
        for i, params in enumerate(model_copy.parameters()):
            params.data = new_params_unfl[i].data
        
        model_copy.to(args.device)
        model_copy.eval()
        
        FLAGS_ = []
        for advs, labels in adv_dataset:
            advs, labels = advs.to(args.device), labels.to(args.device)
            
            logits = model_copy(advs)
            predictions = torch.argmax(logits, dim=-1)
            
            FLAGS_.append((predictions == labels).detach().cpu())
            
        FLAGS.append(torch.stack(FLAGS_))
        
    FLAGS = torch.stack(FLAGS)
    FLAGS = FLAGS.permute((1, 0, 2))
    
    
    ### Reject sampling. Reject non-common adversarial examples ###
    
    final_size = 0
    new_adv_dataset = []
    for i, (advs, labels) in enumerate(adv_dataset):
        f = FLAGS[i]
        # print(f.float().mean())
        
        f = f.all(dim=0)
        # print(f.float().mean())
        
        final_size += f.float().sum()
        if f.float().sum():
            new_adv_dataset.append((advs[f], labels[f]))
        
    print("Final triger set size", final_size.item())
    
    
    ### Evaluate accuracy on stolen models ###
    
    # path_models = glob.glob('./models/resnet50_torchvision' + '/*')
    path_models = glob.glob('./models/resnet34' + '/*')
    
    I_mean = []
    for model_path in path_models:
        # new_model = _models['resnet50'](num_classes=num_classes)
        new_model = resnet34()
        
        new_model.load_state_dict(torch.load(model_path, map_location=args.device))
        new_model.to(args.device)
        new_model.eval()
        
        I_ = []
        for advs, labels in new_adv_dataset:
            advs = advs.to(args.device)
            logits = new_model(advs)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.cpu()
            
            I_.append(predictions == labels)
            
        I_ = torch.cat(I_)
        I_mean.append(I_.float().mean())
        print("Accuracy ", I_.float().mean().item())
        
    I_mean = torch.tensor(I_mean)
    print("Mean acc ", I_mean.mean())
        
        
