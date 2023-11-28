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

from utils import flip_labels, train

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

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs

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
    flip_part = 0.002
    flip_indices = flip_labels(flipset, part=flip_part)
    # Creating subset with flipped indices
    flipset = Subset(flipset, flip_indices)
    # Creating subset with intact indices
    trainset = Subset(trainset, list(set(range(50000)) - set(flipset)))

    models_path = './MODELS/epochs_' + str(num_epochs) + '_size_' + str(flip_part)

    if FLAGS.random_transform:
        models_path += '_randtf'

    models_path += '/models'

    os.makedirs(models_path, exist_ok=True)
    model = resnet34(pretrained=False)

    train_acc_history, flip_acc_history, test_acc_history = train(model, 
                                                                train_set=trainset, 
                                                                flip_set=flipset,
                                                                test_set=testset, 
                                                                num_epochs=num_epochs, 
                                                                batch_size=batch_size, 
                                                                device=device, 
                                                                do_eval=True, 
                                                                epoch_eval=5,
                                                                opt=optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4))

    train_acc, flip_acc, test_acc = train_acc_history[-1], flip_acc_history[-1], test_acc_history[-1]
    print("train accuracy: ", train_acc, ' flip accuracy: ', flip_acc, " test accuracy: ", test_acc)

    PATH = models_path + '/model_watermarking' + '.pt'

    state = {
        "model": model.state_dict(),
        "in_data": flip_indices
    }

    torch.save(state, PATH)

if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 512, 'Batch size')
    flags.DEFINE_integer('num_epochs', 100, 'Training duration in number of epochs.')
    flags.DEFINE_boolean('random_transform', False, 'Using random transform')
    
    app.run(main)
