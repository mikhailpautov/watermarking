import torch
import torchvision
import argparse
import copy

import torchvision.transforms as transforms

from datasets import get_dataset, get_bounds, get_num_classes
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from global_options import _models, _optimizers
from aux import train, accuracy, flatten_params, recover_flattened
from utils import evaluate_model

import foolbox as fb
import matplotlib.pyplot as plt

import glob

from models import resnet18, resnet34, resnet50

class BASE_DATASET(Dataset):
    def __init__(self, args):
        
        self.X = []
        self.y = []
        
        self.train_dataset = get_dataset(args.dataset, 'train')
        self.test_dataset = get_dataset(args.dataset, 'test')
        self.num_classes = get_num_classes(args.dataset)
        self.bounds = get_bounds(args.dataset)
        
        # model = _models[args.model_name](num_classes=num_classes)
        # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        
        self.model = resnet34()
        self.model.load_state_dict(torch.load(args.model_path, map_location='cpu')['model'])
        self.model.eval()
        
        ### Collect adversarial examples ###
        adv_dataset = self.generate()
        
        ### Reject sampling. Compute predictions of proxy models ###
        flatten_dict = flatten_params(self.model.parameters())
        init_params, init_indices = flatten_dict['params'], flatten_dict['indices']    

        FLAGS = []
        model_copy = copy.deepcopy(self.model)
        model_copy.to(args.device)
        
        for i in range(args.M):    
            delta = args.sigma * torch.randn_like(init_params)
            new_params = init_params + delta
            new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
            
            for i, params in enumerate(model_copy.parameters()):
                params.data = new_params_unfl[i].data
            
            model_copy.eval()
            # acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
            
            # while abs(acc_model_copy - acc_model) > 0.05:
            #     delta = args.sigma * torch.randn_like(init_params)
            #     new_params = init_params + delta
            #     new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
                
            #     for i, params in enumerate(model_copy.parameters()):
            #         params.data = new_params_unfl[i].data
                
            #     model_copy.eval()
            #     acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
            #     print(abs(acc_model_copy - acc_model))
            
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
        print(FLAGS.shape)
        final_size = 0
        new_adv_dataset = []
        for i, (advs, labels) in enumerate(adv_dataset):
            f = FLAGS[i]            
            f = f.all(dim=0)
            
            final_size += f.float().sum()
            if f.float().sum():
                self.X.append(advs[f])
                self.y.append(labels[f])
                
        print(final_size)
        print(len(self.y))
        
                
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    def generate(self):
        raise NotImplementedError("Please Implement this method")
        

class ADV_DATASET_(BASE_DATASET):
    def generate(self):
        fmodel = fb.PyTorchModel(self.model, bounds=self.bounds, device=args.device)
        attack = fb.attacks.LinfFastGradientAttack(random_start=True)
        
        epsilons = 0.02  # for adversaarial examples
        I = 0
        adv_dataset = []
        
        # train_dataset or test_dataset ?
        while True:
            idx = torch.randint(0, len(self.test_dataset), (1,)).item()
            image, label = self.test_dataset[idx]
            logits = self.model(image.unsqueeze(dim=0).to(args.device))
            prediction = torch.argmax(logits, dim=-1)
        
            if not label == prediction.item():
                # print("Wrong answer!")
                continue
            
            images = image.repeat(args.batch, 1, 1, 1)
            labels = label * torch.ones(args.batch)
            labels = labels.long()
            images, labels = images.to(args.device), labels.to(args.device)
            
            _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
            
            if not all(success):
                # print("Fail attack")
                continue
            
            logits = self.model(advs)
            target_predictions = torch.argmax(logits, dim=-1)
            
            adv_dataset.append((advs.detach().cpu(), target_predictions.detach().cpu()))
            I += 1

            if I == args.N:
                break
            
        return adv_dataset
    
    
class ADV_DATASET(Dataset):
    def __init__(self, args):
        self.X = []
        self.y = []
        
        t = 0
        while len(self.y) < args.N:
            dataset = ADV_DATASET_(args)
            
            for x_ in dataset.X:
                self.X.append(x_[0])
            for y_ in dataset.y:
                self.y.append(y_[0])
                
            # self.X = self.X + dataset.X
            # self.y = self.y + dataset.y
            
            t += 1
            
        print("ReSample times:", t)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'mnist'], default='cifar10')
    parser.add_argument('--batch', help="batch_size", type=int, default=64)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=5)
    parser.add_argument('--model_name', help = "target model architecture", choices=list(_models.keys()), default='resnet34')
    parser.add_argument('--model_path', help="path to saved model", type=str, default='./model_0.pt')
    parser.add_argument('--N', help="size of trigger set", type=int, default=32)
    
    parser.add_argument('--sigma', help="sigma", type=float, default=1e-4)
    parser.add_argument('--M', help="Number of proxy models", type=int, default=32)
    
    args = parser.parse_args()
    
    adv_dataset = ADV_DATASET(args)
    
    ### Evaluate accuracy on stolen models ###
    
    # path_models = glob.glob('./models/resnet50_torchvision' + '/*')
    path_models = glob.glob('./models/resnet34' + '/*')
    
    adv_loader = DataLoader(adv_dataset, shuffle=False, batch_size=256)
    I_mean = []
    for model_path in path_models:
        # new_model = _models['resnet50'](num_classes=num_classes)
        new_model = resnet34()
        
        new_model.load_state_dict(torch.load(model_path, map_location=args.device))
        new_model.to(args.device)
        new_model.eval()
        
        I_ = []
        
        for advs, labels in adv_loader:
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