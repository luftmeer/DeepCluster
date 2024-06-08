# DeepCluster
Implementation of "Deep Clustering for Unsupervised Learning of Visual Features" by M. Caron, P. Bojanowski, A. Joulin, and M. Douze


# Metrics Measurments
Goal: Test on datasets MNIST and CIFAR10 with the following inputs (for both datasets)
- Model: AlexNet or VGG16
- Input Dimension: 3 (Color), 2 (Color + Sobel), 1 (B/W)
- Classes / k: 10 for both
- Model Optimizer: SGD or Adam (lr= 0.05, momentum=0.9, weight decay=1*10^-5, beta1=0.9, beta2=0.999)
- Top Layer Optimizer: SGD or Adam
- Loss Function Cross Entropy Loss
- Epochs: 100
- Batch Size: 128 (AlexNet) or 32 (VGG16)
- PCA Reductiion: faiss or scikit-learn
- Whitening: faiss or scikit-learn
- Clustering (k-Means): faiss or scikit-learn
- Sobel Filter


## Metrics
### MNIST for all Models (AlexNet or VGG16)
![Metrics for MNIST Dataset](/images/MNIST/MNIST_ALL.png)

> In this overview, a distinctive difference is noticeable in the accuracy. Whereas some models seem to go in a continuously straight line, some models fluctuate heavily by reaching peaks followed by lows and again followed by peaks again which then repeat until the end.
> In regard to the NMI scores, comparing the original True targets with the predicted targets by the clustering algorithm, we notice that a high NMI score is never reached. Instead.
> The NMI scores comparing an epoch with the previous one however shows the authors claims that a certain robustness is reached. However, this simply gives no answer in regard how good the clustering result really is.

#### MNIST for all Models with SGD as main model optimizer
![Metrics for MNIST Dataset with SGD as model optimizer](/images/MNIST/MNIST_optimizer_SGD.png)

#### MNIST for all Models with Adam as main model optimizer
![Metrics for MNIST Dataset with Adam as model optimizer](/images/MNIST/MNIST_optimizer_Adam.png)

#### MNIST for all Models with SGD as top layer optimizer
![Metrics for MNIST Dataset with SGD as model optimizer](/images/MNIST/MNIST_top_layer_optimizer_SGD.png)

#### MNIST for all Models with Adam as top layer optimizer
![Metrics for MNIST Dataset with Adam as model optimizer](/images/MNIST/MNIST_top_layer_optimizer_Adam.png)

### CIFAR10 for all Models (AlexNet or VGG16)
![Metrics for CIFAR10 Dataset](/images/CIFAR10/CIFAR10_ALL.png)

> The results for the CIFAR10 dataset are very contradictory. On the one hand, we reach an accuracy of 1.0 but are unable to determine any valuable NMI score for True and predicted targets.
> On the other hand, the NMI for the epochs show some reasonable results.

#### CIFAR10 for all Models with SGD as main model optimizer
![Metrics for CIFAR10 Dataset with SGD as model optimizer](/images/CIFAR10/CIFAR10_optimizer_SGD.png)

#### CIFAR10 for all Models with Adam as main model optimizer
![Metrics for CIFAR10 Dataset with Adam as model optimizer](/images/CIFAR10/CIFAR10_optimizer_Adam.png)

#### CIFAR10 for all Models with SGD as top layer optimizer
![Metrics for CIFAR10 Dataset with SGD as model optimizer](/images/CIFAR10/CIFAR10_top_layer_optimizer_SGD.png)

#### CIFAR10 for all Models with Adam as top layer optimizer
![Metrics for CIFAR10 Dataset with Adam as model optimizer](/images/CIFAR10/CIFAR10_top_layer_optimizer_Adam.png)