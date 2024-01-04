import torch
import torchvision
import copy

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from global_options import _models, _optimizers
from aux import flatten_params, recover_flattened #train, accuracy,
from utils import evaluate_model

# import foolbox as fb
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

class BASE_DATASET(Dataset):
    def __init__(self, args):
        
        self.X = []
        self.y = []
        
        self.N_base = 32 # Number of watermark images before rejection
        self.args = args
        
        if args.threshold is not None:
            size_ = 512
            test_dataset_for_eval, test_dataset_ = torch.utils.data.random_split(args.test_dataset, [size_, len(args.test_dataset) - size_])
            args.test_dataset = test_dataset_
            dataloader = DataLoader(test_dataset_for_eval, shuffle=False, batch_size=512)
            acc_model = evaluate_model(args.model, dataloader, args.device)
        
        ### Collect adversarial examples ###
        base_dataset = args.train_dataset if args.use_train else args.test_dataset
        adv_dataset = self.generate(base_dataset)
        
        ### Reject bad watermarks ###
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
            
            if args.threshold is not None:
                acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
                
                while abs(acc_model_copy - acc_model) > args.threshold:
                    delta = args.sigma * torch.randn_like(init_params)
                    new_params = init_params + delta
                    new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
                    
                    for i, params in enumerate(model_copy.parameters()):
                        params.data = new_params_unfl[i].data
                    
                    model_copy.eval()
                    acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
                    print("Accuracy gap:", abs(acc_model_copy - acc_model))
            
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

        # print("final size", final_size)
        print("len dataset", len(self.y))
        print("reject coefficient", len(self.y) / self.N_base)
        
                
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    def generate(self):
        raise NotImplementedError("Please Implement this method")
        

class ADV_DATASET_(BASE_DATASET):
    def generate(self, base_dataset):
        args = self.args
        fmodel = fb.PyTorchModel(args.model, bounds=args.bounds, device=args.device)
        attack = fb.attacks.LinfFastGradientAttack(random_start=True)
        
        epsilons = 0.02  # for adversaarial examples
        I = 0
        adv_dataset = []
        
        while True:
            idx = torch.randint(0, len(base_dataset), (1,)).item()
            image, label = base_dataset[idx]
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

            if I == self.N_base:
                break
            
        return adv_dataset
        
        
class L_DATASET_(BASE_DATASET):
    def __init__(self, args, idxs=[]):
        self.idxs = idxs
        super().__init__(args)
    
    
    def generate(self, base_dataset):
        args = self.args
        I = 0
        adv_dataset = []
        
        while True:
            idx1 = torch.randint(0, len(base_dataset), (1,)).item()
            idx2 = torch.randint(0, len(base_dataset), (1,)).item()

            if idx1 == idx2:
                continue
            
            if (idx1 in self.idxs) or (idx2 in self.idxs):
                # print("wau")
                continue
            
            image1, label1 = base_dataset[idx1]
            image2, label2 = base_dataset[idx2]
            
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
                self.idxs.append(idx1)
                self.idxs.append(idx2)
                
            
            if I == self.N_base:
                break
            
        return adv_dataset


class FINAL_DATASET(Dataset):
    def __init__(self, args):
        self.X = []
        self.y = []
        
        times = []
        t = 0
        idxs = []
        while len(self.y) < args.N:
            start = time.time()
            dataset = L_DATASET_(args, idxs)
            idxs = dataset.idxs
            end = time.time()
            times.append(end - start)
            
            # TODO add another selection criterion
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