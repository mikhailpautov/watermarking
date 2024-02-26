from absl import app, flags
import torch
from datasets import get_dataset, get_num_classes
from distillation_extraction import steal
from basic_train import train_model
from global_options import _models

# Define flags
flags.DEFINE_integer('num_epochs', 100, 'Training duration in number of epochs.')
flags.DEFINE_integer('num_models', 10, 'Amount of models to be trained')
flags.DEFINE_boolean('random_transform', False, 'Using random transform')
flags.DEFINE_string('save_path', './models/resnet18/', 'Path to save model')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset for model stealing')
flags.DEFINE_string('stealing_dataset', 'cifar10', 'Dataset on which stealing will be performed')
flags.DEFINE_string('student_name', 'resnet34', 'Student model architecture')
flags.DEFINE_string('teacher_name', 'resnet34', 'Target model architecture')
flags.DEFINE_string('teacher_path', './teacher_cifar10_resnet34/model_1', 'Path to Teacher model')
flags.DEFINE_boolean('do_eval', True, 'Evaluating student model during distillation')
flags.DEFINE_integer('epoch_eval', 10, 'How often eval should be performed')
flags.DEFINE_integer('save_iter', 5, 'How often model should be saved')
flags.DEFINE_float('temperature', 1.0, 'The Temperature hyperparameter for stealing attacks')
flags.DEFINE_float('alpha', 0.3, 'Alpha hyperparameter for RGT stealing attacks')
flags.DEFINE_string('policy', 'soft', 'Stealing policy')
flags.DEFINE_float('stop_acc', 0.99, 'Early stop if student got sufficient accuracy')

FLAGS = flags.FLAGS

def main(argv):
    del argv  # Unused

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parameters from flags
    num_epochs = FLAGS.num_epochs
    num_models = FLAGS.num_models
    save_path = FLAGS.save_path
    dataset = FLAGS.dataset
    stealing_dataset = FLAGS.stealing_dataset
    teacher_name = FLAGS.teacher_name
    teacher_path = FLAGS.teacher_path
    student_name = FLAGS.student_name
    do_eval = FLAGS.do_eval
    epoch_eval = FLAGS.epoch_eval
    save_iter = FLAGS.save_iter
    T = FLAGS.temperature
    alpha = FLAGS.alpha
    policy = FLAGS.policy
    stop_acc = FLAGS.stop_acc
    num_classes = get_num_classes(dataset)

    # Load teacher model
    teacher_model = _models[teacher_name](pretrained=False, num_classes=num_classes)
    teacher_model.load_state_dict(torch.load(teacher_path))

    # Distillation
    for i in range(num_models):
        student_model = _models[student_name](pretrained=False, num_classes=num_classes)
        steal(student_name, teacher_model, stealing_dataset, dataset, num_epochs, device,
              do_eval, epoch_eval, save_path, save_iter, T, alpha, policy, stop_acc)

if __name__ == '__main__':
    app.run(main)
