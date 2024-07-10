### FashionMNIST
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


![image](../images/FashionMNIST_Overview_AlexNet_VGG16_Sobel.png)


### Commands for easy repetition
#### AlexNet
**Without Sobel**: (dacit)
```bash
for i in {1..3}; do python3 main.py --arch AlexNet --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; done
```

**With Sobel**:
```bash
for i in {1..3}; do python3 main.py --arch AlexNet --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch AlexNet --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; done
```

#### VGG16
**Without Sobel**:
[//]: # (adakit)
```bash
for i in {1..5}; do python3 main.py --arch VGG16 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; done
```

**With Sobel**:
[//]: # (charnockit)
```bash
for i in {1..5}; do python3 main.py --arch VGG16 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch VGG16 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 32 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; done
```

---

#### ResNet18
**Without Sobel**: (dacit)
```bash
for i in {1..3}; do python3 main.py --arch ResNet18 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 1 --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; done
```

**With Sobel**: (biolithe)
```bash
for i in {1..3}; do python3 main.py --arch ResNet18 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering faiss --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method faiss --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; python3 main.py --arch ResNet18 --input_dim 2 --sobel --num_classes 10 --epochs 25 --requires_grad --dataset FashionMNIST --ds_train --batch_size 128 --reassign_optimizer_tl --pca --pca_method sklearn --pca_whitening --clustering sklearn --metrics --metrics_dir ./metrics/FashionMNIST/5_runs_100_epochs/ -v; done
```