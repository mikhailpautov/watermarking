# Watermarking
(Robust) watermarking

## Commands
Create watermarking and test on saved models:

```python create_test_watermarks.py --dataset cifar10 --model_name resnet34 --model_path ./teacher_cifar10_resnet34/model_1 --models_name vgg11 --models_path ./stealing_vgg11_cifar10_soft --sigma1 8e-3 --M 64 --N 100```

Available settings:
1. You can use train split instead of test by changing ``--use_train`` to True.
2. You can control the maximum accuracy deviation of proxy models using a "threshold".

