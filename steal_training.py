from typing import Any

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import torch
import torchvision.datasets as datasets

from models import resnet34

from utils import steal_train, models_mse, get_dataset

from absl import app, flags

FLAGS = flags.FLAGS
def main(argv):
    del argv

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    num_models = FLAGS.num_models
    save_path = FLAGS.save_path
    dataset = FLAGS.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available Device ', device)

    trainset, testset = get_dataset(dataset)
    model_path = FLAGS.target_path
    model = resnet34(pretrained=False)

    data = torch.load(model_path)
    model.load_state_dict(data['model'])

    flip_indexes = data['in_data']

    for i in range(num_models):
        temp_model = resnet34(pretrained=False)
        steal_train(model, temp_model, trainset, testset, num_epochs, device, 
                    do_eval=True, epoch_eval=5, batch_size=batch_size)
        mse_value = models_mse(temp_model, model)
        torch.save(temp_model.state_dict(), save_path + 'model_mse_' + str(round(mse_value, 3)))

if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 512, 'Batch size')
    flags.DEFINE_integer('num_epochs', 100, 'Training duration in number of epochs.')
    flags.DEFINE_integer('num_models', 100, 'Amount of models to be trained')
    flags.DEFINE_boolean('random_transform', False, 'Using random transform')
    flags.DEFINE_string('target_path', '/workspace/watermarking/MODELS/epochs_100_size_0.002/models/model_watermarking.pt', 'Path to target model')
    flags.DEFINE_string('save_path', './models/', 'Path to save model')
    flags.DEFINE_string('dataset', 'cifar100', 'Dataset for model stealing')    
    
    app.run(main)