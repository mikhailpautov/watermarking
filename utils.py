from copy import deepcopy
import os

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import Trainer


def get_train_splits_for_models(D_train_aux, D_test_aux, num_models):    
    
    n = len(D_train_aux)
    perm_indices = np.random.permutation(n)
    m = n // num_models
    
    train_splits = []
    
    for j in range(num_models):
        train_subset = Subset(D_train_aux, perm_indices[j*m:(j+1)*m])
        train_splits.append(train_subset)
    return train_splits, D_test_aux


def get_rand_train_splits_for_models(D_train_aux, D_test_aux, num_models, set_size):
    n = len(D_train_aux)
    train_splits = []
    
    for _ in range(num_models):
        perm_indices = np.random.permutation(n)
        inxs = perm_indices[:set_size]
        train_subset = Subset(D_train_aux, inxs)
        train_splits.append(train_subset)
    return train_splits, D_test_aux

            
def train_in_out_models(model, trainset, testset, n_models, set_size, training_args, save_path='./models'):
    train_splits, D_test_aux = get_rand_train_splits_for_models(trainset, testset, n_models, set_size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for j in tqdm(range(n_models)):
        temp_model = deepcopy(model)
        train_j = train_splits[j]
        trainer = Trainer(model=temp_model,
                          args=training_args,
                          train_dataset=train_j,
                          eval_dataset=testset)
        
        trainer.train()
        state = {
            'model':temp_model.state_dict(),
            'in_data':train_j,
            'out_data': np.array(list(set(range(len(trainset))) - set(train_j.indices)))
        }
        torch.save(state, save_path + f'/data_{j}')


def tokenize_function(sentence, tokenizer):
    return tokenizer(sentence['text'], padding='max_length', truncation=True, max_length=512)


def create_subset(dataset, subset_size=5000):
    indices = np.random.permutation(len(dataset))[:subset_size]
    return dataset.select(indices)
