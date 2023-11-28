from typing import Any

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import foolbox as fb
import matplotlib.pyplot as plt


from torchvision.models.resnet import ResNet34_Weights
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Subset, DataLoader, Dataset
from models import resnet34
import torchvision.transforms as transforms

from utils import flip_labels, train, evaluate_model, add_uniform_noise
from copy import deepcopy

from absl import app, flags

FLAGS = flags.FLAGS

transform_train_ = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

transform_test_ = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
])


def main(argv):
    """
    Creates n_models models
    Trains these models on subset of train with size set_size
    Train parameters: num_epochs, batch_size

    If check_test is True then computes accuracy on test and saves model if accuracy >= threshold

    Path to models and indexes of train subset
    ./MODELS/epochs_***_size_***_acc_***_randtf/models/model_***.pt

    """
    del argv
    
    n_models = FLAGS.n_models
    eps = FLAGS.eps
    data_path = FLAGS.data_path

    data = torch.load(data_path)

    if FLAGS.random_transform:
        transform_train = transform_train_
        transform_test = transform_test_
    else:
        transform_train = transform
        transform_test = transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    flipset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Flip part of the datset
    flip_indices = data['in_data']
    # Creating subset with flipped indices
    flipset = Subset(flipset, flip_indices)
    # Creating subset with intact indices
    trainset = Subset(trainset, list(set(range(50000)) - set(flipset)))

    model = resnet34(pretrained=False)
    model.load_state_dict(data['model'])
    model.to(device)

    trainloader = DataLoader(trainset, batch_size=512, shuffle=True)
    fliploader = DataLoader(flipset, batch_size=512, shuffle=True)
    testloader = DataLoader(testset, batch_size=512, shuffle=True)



    epses = [ 0.001, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.03]
    train_acces = []
    flip_acces = []
    test_acces = []

    for eps in epses:

        train_acc = []
        flip_acc = []
        test_acc = []
        for i in tqdm(range(n_models)):
            temp_model = deepcopy(model)
            add_uniform_noise(temp_model, eps)

            # _train_acc = evaluate_model(model, trainloader, device)
            _flip_acc = evaluate_model(temp_model, fliploader, device)
            _test_acc = evaluate_model(temp_model, testloader, device)
            # train_acc.append(_train_acc)
            flip_acc.append(_flip_acc)
            test_acc.append(_test_acc)

        # train_acc = np.array(train_acc)
        flip_acc = np.array(flip_acc)
        test_acc = np.array(test_acc)
        flip_acces.append(np.mean(flip_acc))
        test_acces.append(np.mean(test_acc))
    save_data = {'flip_acces':np.array(flip_acces),
                 'etst_acces':np.array(test_acc),
                 'epses':np.array(epses)}
    np.savez('data.npz', save_data)
    # Saving plots 
    # Plot for flip_acces
    plt.figure(figsize=(8, 6))
    plt.plot(epses, flip_acces, marker='o', color='b', label='Flip Accuracy')
    plt.xlabel('Epses')
    plt.ylabel('Flip Accuracy')
    plt.title('Flip Accuracy vs Epsillons')
    plt.legend()
    plt.grid(True)
    plt.savefig('flip_accuracy_plot.png')

    # Plot for test_acces
    plt.figure(figsize=(8, 6))
    plt.plot(epses, test_acces, marker='o', color='r', label='Test Accuracy')
    plt.xlabel('Epses')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Epsillons')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_accuracy_plot.png')
    
    # print(f'Train acc {np.mean(train_acc):.3f}, Flip acc {np.mean(flip_acc):.3f}, Test acc {np.mean(test_acc):.3f}')
    # print(f'Flip acc {np.mean(flip_acc):.3f}, Test acc {np.mean(test_acc):.3f}')

if __name__ == '__main__':
    flags.DEFINE_string('data_path', '/workspace/watermarking/MODELS/epochs_100_size_0.002/models/model_watermarking.pt', 'Path to model')
    flags.DEFINE_boolean('random_transform', False, 'Using random transform')
    flags.DEFINE_float('eps', 0.0, 'Uniform noise to model weight const')
    flags.DEFINE_integer('n_models', 64, 'Define the number of models ')
    app.run(main)
