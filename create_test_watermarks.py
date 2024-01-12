import torch
import argparse
import glob
import copy

from torch.utils.data import DataLoader

from datasets import get_dataset, get_bounds, get_num_classes
from global_options import _models, _optimizers
from watermark_dataset import FINAL_DATASET
from aux import flatten_params, recover_flattened
    
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'mnist'], default='cifar10')
    parser.add_argument('--batch', help="batch_size", type=int, default=64)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=0)
    parser.add_argument('--model_name', help = "target model architecture", choices=list(_models.keys()), default='resnet34')
    parser.add_argument('--model_path', help="path to saved model", type=str, default='./model_0.pt')
    parser.add_argument('--models_name', help = "stolen model architecture", choices=list(_models.keys()), default='resnet18')
    parser.add_argument('--models_path', help="path to stolen models", type=str, default='./models/resnet18')
    parser.add_argument('--N', help="size of trigger set", type=int, default=100)
    parser.add_argument('--sigma', help="sigma", type=float, default=1e-3)
    parser.add_argument('--M', help="number of proxy models", type=int, default=32)
    parser.add_argument('--threshold', help="threshold for proxy models", type=float, default=None)
    parser.add_argument('--use_train', help="watermarks based on train?", type=bool, default=False)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.use_train:
        args.train_dataset = get_dataset(args.dataset, 'train')
    args.test_dataset = get_dataset(args.dataset, 'test')
    args.num_classes = get_num_classes(args.dataset)
    args.bounds = get_bounds(args.dataset)
    
    model = _models[args.model_name](num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(args.device)
    model.eval()
    
    args.model = model
    
    adv_dataset = FINAL_DATASET(args)
    adv_loader = DataLoader(adv_dataset, shuffle=False, batch_size=256)
    
    ### Evaluate accuracy on stolen models ###
    print("Start evaluate stolen models")
    path_models = glob.glob(args.models_path + '/*')
    
    new_model = _models[args.models_name](num_classes=args.num_classes)
    
    I_mean = []
    for model_path in path_models:
        new_model.load_state_dict(torch.load(model_path, map_location=args.device))
        new_model.to(args.device)
        new_model.eval()
        
        I_ = []
        
        for data in adv_loader:
            advs, labels = data[0], data[1]
            advs = advs.to(args.device)
            logits = new_model(advs)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.cpu()
            
            I_.append(predictions == labels)
            
        I_ = torch.cat(I_)
        I_mean.append(I_.float().mean())
        print("Accuracy ", I_.float().mean().item())
        
    I_mean = torch.tensor(I_mean)
    print("======================")
    print(f"Mean acc {I_mean.mean().item():.3f} +- {I_mean.std().item():.3f}")
    

    ### Evaluate accuracy on proxy models ###
    print("Start evaluate proxy models")
    flatten_dict = flatten_params(args.model.parameters())
    init_params, init_indices = flatten_dict['params'], flatten_dict['indices']    

    model_copy = copy.deepcopy(args.model)
    model_copy.to(args.device)
    model_copy.eval()
    
    I_mean = []
    for i in range(len(path_models)):  
        delta = args.sigma * torch.randn_like(init_params)
        new_params = init_params + delta
        new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
        
        for i, params in enumerate(model_copy.parameters()):
            params.data = new_params_unfl[i].data
        
        I_ = []
        
        for data in adv_loader:
            advs, labels = data[0], data[1]
            advs = advs.to(args.device)
            logits = model_copy(advs)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.cpu()
            
            I_.append(predictions == labels)
            
        I_ = torch.cat(I_)
        I_mean.append(I_.float().mean())
        print("Accuracy ", I_.float().mean().item())
        
    I_mean = torch.tensor(I_mean)
    print("======================")
    print(f"Mean acc {I_mean.mean().item():.3f} +- {I_mean.std().item():.3f}")