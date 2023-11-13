from torch.optim import SGD, Adam
from torchvision.models import resnet18, resnet34

_optimizers = {
        'SGD': SGD, 
          'Adam':Adam
          }

_models = {
    'resnet18': resnet18,
    'resnet34': resnet34}