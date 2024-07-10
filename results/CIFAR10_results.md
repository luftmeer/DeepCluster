### CIFAR10
- Tested with faiss and scikit-learn library for both PCA and k-Means clustering with each possible combination
- Plots:
    1. Average loss for each epoch
    2. Training accuracy for pseudo-labels and output
    3. True accuracy for true labels and output
    4. NMI tested on the true labels and the pseudo-labels
    5. NMI tested on the clustering results of the current and previous epoch (at first epoch always 0.0)
    6. Time take for each segment and total (blue) -> TODO: Coloring
- Colors:
  - blue: Sobel filtering is **not** active
  - green: Sobel filtering is active


![image](../images/CIFAR10_Overview_AlexNet_VGG16_Sobel.png)


#### AlexNet
**Without Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch AlexNet --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```

**With Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```

#### VGG16
**Without Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch VGG16 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```

**With Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch VGG16 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```
---

#### AlexNet (Adjusted Learning Rate (0.1), Weight Decay (0.001), num_clusters (1000), batch size (64))
**Without Sobel**:

```bash
for i in {1..3}; do python3 main.py --arch AlexNet --input_dim 3 --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; python3 main.py --arch AlexNet --input_dim 3 --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; python3 main.py --arch AlexNet --input_dim 3 --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; python3 main.py --arch AlexNet --input_dim 3 --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; done
```

**With Sobel**:
```bash
for i in {1..3}; do python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; python3 main.py --arch AlexNet --input_dim 2 --grayscale --sobel --num_classes 1000 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 64 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v --lr 0.1 --weight_decay 0.001 --lr_tl 0.1 --weight_decay_tl 0.001 ; done
```

#### ResNet18
**Without Sobel**:
```bash  (adakit)
for i in {1..5}; do python3 main.py --arch ResNet18 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```

**With Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch ResNet18 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```

#### FeedForward (Without PCA due to lower Feature Space)
**Without Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch FeedForward --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch FeedForward --input_dim 3 --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```

**With Sobel**:
```bash
for i in {1..5}; do python3 main.py --arch FeedForward --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --clustering faiss --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; python3 main.py --arch FeedForward --input_dim 2 --grayscale --sobel --num_classes 10 --epochs 25 --requires_grad --dataset CIFAR10 --ds_train --batch_size 128 --reassign_optimizer_tl --clustering sklearn --metrics --metrics_dir ./metrics/CIFAR10/5_runs_100_epochs/ -v; done
```