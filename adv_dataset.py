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
import time
from tqdm import tqdm

from models import resnet18, resnet34, resnet50

class BASE_DATASET(Dataset):
    def __init__(self, args):
        
        self.X = []
        self.y = []
        
        self.models_batch = 10
        self.args = args
        
        # size_ = 512
        # test_dataset_,  _= torch.utils.data.random_split(args.test_dataset, [size_, len(args.test_dataset) - size_])
        # dataloader = DataLoader(test_dataset_, shuffle=False, batch_size=512)
        # acc_model = evaluate_model(args.model, dataloader, args.device)
        
        ### Collect adversarial examples ###
        adv_dataset = self.generate()
        
        ### Reject sampling. Compute predictions of proxy models ###
        flatten_dict = flatten_params(args.model.parameters())
        init_params, init_indices = flatten_dict['params'], flatten_dict['indices']    

        model_copy = copy.deepcopy(args.model)
        model_copy.to(args.device)
        
        for i in tqdm(range(args.M)):  
            delta = args.sigma * torch.randn_like(init_params)
            new_params = init_params + delta
            new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
            
            for i, params in enumerate(model_copy.parameters()):
                params.data = new_params_unfl[i].data
            
            model_copy.eval()
            # acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
            
            # while abs(acc_model_copy - acc_model) > 0.1:
            #     delta = args.sigma * torch.randn_like(init_params)
            #     new_params = init_params + delta
            #     new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
                
            #     for i, params in enumerate(model_copy.parameters()):
            #         params.data = new_params_unfl[i].data
                
            #     model_copy.eval()
            #     acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
            #     print(abs(acc_model_copy - acc_model))
            
            update_adv_dataset = []
            for advs, labels in adv_dataset:
                advs, labels = advs.to(args.device), labels.to(args.device)
                
                logits = model_copy(advs)
                predictions = torch.argmax(logits, dim=-1)
                
                f = (predictions == labels).detach().cpu()
                if f.float().sum():
                    update_adv_dataset.append((advs[f], labels[f]))
                    
            adv_dataset = update_adv_dataset
            
        
        final_size = 0
                
        for i, (advs, labels) in enumerate(adv_dataset):
            final_size += labels.shape[0]
            self.X.append(advs.detach().cpu())
            self.y.append(labels.detach().cpu())

        print("final size", final_size)
        print("len dataset", len(self.y))
        
                
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    def generate(self):
        raise NotImplementedError("Please Implement this method")
        

class ADV_DATASET_(BASE_DATASET):
    def generate(self):
        args = self.args
        fmodel = fb.PyTorchModel(args.model, bounds=args.bounds, device=args.device)
        attack = fb.attacks.LinfFastGradientAttack(random_start=True)
        
        epsilons = 0.02  # for adversaarial examples
        I = 0
        adv_dataset = []
        
        # train_dataset or test_dataset ?
        while True:
            idx = torch.randint(0, len(args.test_dataset), (1,)).item()
            image, label = args.test_dataset[idx]
            logits = args.model(image.unsqueeze(dim=0).to(args.device))
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
            
            logits = args.model(advs)
            target_predictions = torch.argmax(logits, dim=-1)
            
            adv_dataset.append((advs.detach().cpu(), target_predictions.detach().cpu()))
            I += 1

            if I == self.models_batch:
                break
            
        return adv_dataset
        
        
class L_DATASET_(BASE_DATASET):
    def generate(self):
        args = self.args
        I = 0
        adv_dataset = []
        
        while True:
            idx1 = torch.randint(0, len(args.test_dataset), (1,)).item()
            idx2 = torch.randint(0, len(args.test_dataset), (1,)).item()

            if idx1 == idx2:
                continue
            
            image1, label1 = args.test_dataset[idx1]
            image2, label2 = args.test_dataset[idx2]
            
            if label1 == label2:
                continue
            
            image1 = image1.repeat(args.batch, 1, 1, 1)
            image2 = image2.repeat(args.batch, 1, 1, 1)
            
            lambda_ = torch.rand(args.batch, 1, 1, 1)
            images = (1 - lambda_) * image1 + lambda_ * image2
            images = images.to(args.device)
            
            logits = args.model(images)
            target_predictions = torch.argmax(logits, dim=-1)
            
            images, target_predictions = images.detach().cpu(), target_predictions.detach().cpu()
            f = (target_predictions != label1) * (target_predictions != label2)
            
            if f.float().sum():
                images, target_predictions = images[f], target_predictions[f]
                adv_dataset.append((images, target_predictions))
                I += 1
            
            if I == self.models_batch:
                break
            
        return adv_dataset


class FINAL_DATASET(Dataset):
    def __init__(self, args, class_dataset):
        self.X = []
        self.y = []
        
        times = []
        t = 0
        while len(self.y) < args.N:
            start = time.time()
            dataset = class_dataset(args)
            end = time.time()
            times.append(end - start)
            
            for x_ in dataset.X:
                self.X.append(x_[0])
                if len(self.X) == args.N:
                    break
            for y_ in dataset.y:
                self.y.append(y_[0])
                if len(self.y) == args.N:
                    break
            
            t += 1
            
        print("ReSample times:", t)
        print("Average time:", sum(times) / len(times))
        
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
    parser.add_argument('--N', help="size of trigger set", type=int, default=100)
    
    parser.add_argument('--sigma', help="sigma", type=float, default=1e-3)
    parser.add_argument('--M', help="Number of proxy models", type=int, default=32)
    
    args = parser.parse_args()
    
    args.train_dataset = get_dataset(args.dataset, 'train')
    args.test_dataset = get_dataset(args.dataset, 'test')
    args.num_classes = get_num_classes(args.dataset)
    args.bounds = get_bounds(args.dataset)
    
    model = _models[args.model_name]()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu')['model'])
    model.to(args.device)
    model.eval()
    
    args.model = model
    
    adv_dataset = FINAL_DATASET(args, L_DATASET_)
    print(len(adv_dataset))
    
    ### Evaluate accuracy on stolen models ###
    print("Start evaluate stolen models")
    path_models = glob.glob('./models/resnet18' + '/*')
    
    adv_loader = DataLoader(adv_dataset, shuffle=False, batch_size=256)
    I_mean = []
    for model_path in path_models:
        new_model = _models['resnet18']()
        
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
    

    ### Evaluate accuracy on proxy models ###
    print("Start evaluate proxy models")
    flatten_dict = flatten_params(args.model.parameters())
    init_params, init_indices = flatten_dict['params'], flatten_dict['indices']    

    model_copy = copy.deepcopy(args.model)
    model_copy.to(args.device)
    
    I_mean = []
    for i in range(len(path_models)):  
        delta = args.sigma * torch.randn_like(init_params)
        new_params = init_params + delta
        new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
        
        for i, params in enumerate(model_copy.parameters()):
            params.data = new_params_unfl[i].data
        
        model_copy.eval()
        
        I_ = []
        
        for advs, labels in adv_loader:
            advs = advs.to(args.device)
            logits = model_copy(advs)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.cpu()
            
            I_.append(predictions == labels)
            
        I_ = torch.cat(I_)
        I_mean.append(I_.float().mean())
        print("Accuracy ", I_.float().mean().item())
        
    I_mean = torch.tensor(I_mean)
    print("Mean acc ", I_mean.mean())