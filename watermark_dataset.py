import torch
import torchvision
import copy

import torchvision.transforms as transforms

from torch.distributions import Beta
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from global_options import _models, _optimizers
from aux import flatten_params, recover_flattened
from utils import evaluate_model

# import foolbox as fb
import matplotlib.pyplot as plt

import time
from tqdm import tqdm


def reject_data(data, f):
    return tuple(data[i][f] for i in range(len(data)))
        
def data_to_device(data, device='cpu'):
    return tuple(data[i].detach().to(device) for i in range(len(data)))
        
    
class BASE_DATASET(Dataset):
    def __init__(self, args):
        
        self.data = []
        
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
            for data in adv_dataset:
                advs, labels = data[0], data[1]
                advs, labels = advs.to(args.device), labels.to(args.device)
                
                logits = model_copy(advs)
                predictions = torch.argmax(logits, dim=-1)
                
                f = (predictions == labels).detach().cpu()
                if f.float().sum():
                    update_adv_dataset.append(reject_data(data, f))
                    
            adv_dataset = update_adv_dataset
            
                
        for data in adv_dataset:
            self.data.append(data_to_device(data))

        print("len dataset", len(self.data))
        print("reject coefficient", len(self.data) / self.N_base)
        
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
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
            
            if (idx1 in self.idxs) and (idx2 in self.idxs):
                continue
            
            image1, label1 = base_dataset[idx1]
            image2, label2 = base_dataset[idx2]
            
            if label1 == label2:
                continue
            
            logits1 = args.model(image1.unsqueeze(dim=0).to(args.device)).detach()
            logits2 = args.model(image2.unsqueeze(dim=0).to(args.device)).detach()
            predictions1 = torch.argmax(logits1, dim=-1)
            predictions2 = torch.argmax(logits2, dim=-1)
            
            if (predictions1 != label1) or (predictions2 != label2):
                continue
            
            image1 = image1.repeat(args.batch, 1, 1, 1)
            image2 = image2.repeat(args.batch, 1, 1, 1)
            
            # lambda_ = torch.rand(args.batch, 1, 1, 1)
            alpha = 0.1
            m = Beta(alpha, alpha)
            lambda_ = m.sample((args.batch, 1, 1, 1))
            
            images = (1 - lambda_) * image1 + lambda_ * image2
            images = images.to(args.device)
            lambda_ = lambda_.squeeze()
            
            logits = args.model(images).detach()

            target_predictions = torch.argmax(logits, dim=-1)
            
            images, target_predictions = images.detach().cpu(), target_predictions.detach().cpu()
            f = (target_predictions != label1) * (target_predictions != label2)
            
            if f.float().sum():
                images, target_predictions = images[f], target_predictions[f]
                lambda_ = lambda_[f]

                adv_dataset.append((images, target_predictions, lambda_))
                I += 1
                self.idxs.append(idx1)
                self.idxs.append(idx2)

                # print(label1, label2)
                # print(torch.unique(target_predictions, return_counts=True))
                
            if I == self.N_base:
                break
            
        return adv_dataset


class FINAL_DATASET(Dataset):
    def __init__(self, args):
        self.data = []
        
        times = []
        t = 0
        idxs = []
        while len(self.data) < args.N:
            start = time.time()
            dataset = L_DATASET_(args, idxs)
            idxs = dataset.idxs
            end = time.time()
            times.append(end - start)
            
            # TODO add another selection criterion
            for data in dataset:
                # self.data.append(tuple(data[i][0] for i in range(len(data))))
                
                lambda_ = data[2]
                lambda_ = torch.abs(lambda_ - 0.5)
                idx = torch.argmax(lambda_)
                self.data.append(tuple(data[i][idx] for i in range(len(data))))
                               
                if len(self.data) == args.N:
                    break
            print("Elements in trigerset:", len(self.data))  
            t += 1
            
        print("ReSample times:", t)
        print("Average time:", sum(times) / len(times))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]