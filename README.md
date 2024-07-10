# An implementation and adjustment of DeepCluster

"**Deep Clustering for Unsupervised Learning of Visual Features**" by M. Caron, P. Bojanowski, A. Joulin, and M. Douze ([arXiv](https://arxiv.org/abs/1807.05520) [[PDF](https://arxiv.org/pdf/1807.05520)])
See original code: https://github.com/facebookresearch/deepcluster/

---

## Base idea of DeepCluster

DeepCluster is a method that jointly learn the parameters of a neural network and the cluster assignments of the resulting features. By iteratively grouping the features with a standard clustering algorithm (k-Means), it uses the subsequent assignments as supervision to update the weights of the network.

---

## Adjustments & Improvements

#### PCA & Clustering

##### Algorithms

In their original implementation, the authors used a different implementation for their PCA reduction as well for their k-Means approach. The implementations are based on the [faiss](https://github.com/facebookresearch/faiss) package by Facebook, Inc.
The PCA version of **faiss** has no special advancement as basis, whereas the k-Means implementation is based on "Billion-scale similarity search with gpus" by Johnson et. al ([arXiv](https://arxiv.org/abs/1702.08734) [[PDF](https://arxiv.org/pdf/1702.08734)]) which is optimized for GPU usage and should outperform other k-Means approaches.

Our adjustment is to simply replace the authors choices with commonly used implementations by the [scikit-learn](https://scikit-learn.org) library.

##### Merging input images with pseudo-labels

When merging the input images and their respective clustered pseudo-labels, the authors created a custom PyTorch Dataset object. For the **faiss** implementation, we [reuse](/deepcluster/utils/faiss_kmeans.py) their approach as well.

For the **scikit-learn** implementation, another special [custom dataset](/deepcluster/utils/pseudo_labeled_dataset.py) is added. We simply store the original dataset (with true targets) and the pseudo-labels as class attributes. When the train loader tries to fetch images, we split the dataset to image and true target and return a transformed image with its respective pseudo-label and the true target.

The true target is necessary in both cases, since we also want to calculate the true accuracy and not just the accuracy for pseudo-labels and the output result. The model however is only trained on the pseudo-label targets.

#### Contrastive Learning

We have tried to increase the performance of DeepCluster by combining it with Contrastive Learning.
Contrastive learning is a technique from machine learning in which the representations of the images in the latent space are "pushed towards" images of the same class and "pushed away" from images of other classes.

We have implemented two strategies for this:

##### Contrastive Strategy 1

For the first strategy we feed the input image into the DeepCluster model in order to get features and pseudo labels.
These resulting features and labels we feed into a contrastive head, that computes the Contrastive Loss, which we will add to the DeepCluster Loss obtained from the DeepCluster model.

Here you can see a figure of the strategy:

<p align="center">
  <img src="https://github.com/luftmeer/DeepCluster/assets/76487137/6fd406fb-59aa-45a3-8356-18565660d687" height=500 alt="Contrastive Straategy 1"/>
</p>

##### Contrastive Strategy 2

The first strategy relies on the computed pseudo labels, which is not optimal, since they can be wrong.
Therefore we utilize image augmentation to create two different images from one single image. Since both come from the same origin, we know, that both representations should be in the same cluster, i.e. closer together.

We feed both images separately into the DeepCluster framework and extract the resulting representations of the augmented images.
These representations we then feed into our contrastive head, that calculates the NTXent Loss and returns it.
Together with the two DeepCluster losses the sum of the losses forms the final loss.

Here is a figure of the process:

<p align="center">
  <img src="https://github.com/luftmeer/DeepCluster/assets/76487137/500fb848-1d82-43ea-99cd-2fb69c3625c5" height=500 alt="Contrastive Straategy 2"/>
</p>

---

## Metrics

For results on datasets, see:

- [MNIST](/results/MNIST_results.md)
- [CIFAR10](/results/CIFAR10_results.md)
- [FashionMNIST](/results/FashionMNIST_results.md)
- [GTSRB](/results/GTSRB_results.md)

---

## Usage

The Model can simply be run by using the [main.py](/main.py) file in this root directoy. See the following help overview on how to use each individual input flat.

> **_NOTE_**: Some of the inputs are not necessary and were strictly used for testing different values or approaches. Simply executing `python3 main.py`will result in a working run, as long a Nvidia GPU is present. If no GPU is present, use `python3 main.py --clustering sklearn` since the **faiss** clustering approach strictly needs a GPU present.

```text
usage: python3 main.py [-h]
                       [--arch {FeedForward,AlexNet,VGG16,ResNet18,ResNet34,ResNet50,ResNet101,ResNet152}]
                       [--input_dim INPUT_DIM] [--num_classes NUM_CLASSES] [--sobel] [--grayscale]
                       [--requires_grad] [--epochs EPOCHS]
                       [--dataset {CIFAR10,MNIST,FashionMNIST,KMNIST,USPS,tinyimagenet,STL10,GTSRB}]
                       [--data_dir DATA_DIR] [--ds_train]
                       [--ds_split {train,test,unlabeled,train+unlabeled,val}]
                       [--batch_size BATCH_SIZE] [--optimizer {SGD,Adam}] [--lr LR]
                       [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--beta1 BETA1]
                       [--beta2 BETA2] [--param_requires_grad] [--reassign_optimizer_tl]
                       [--optimizer_tl {SGD,Adam}] [--lr_tl LR_TL] [--momentum_tl MOMENTUM_TL]
                       [--weight_decay_tl WEIGHT_DECAY_TL] [--beta1_tl BETA1_TL]
                       [--beta2_tl BETA2_TL] [--loss_fn {L1,L2,MSE,CrossEntropy}] [--pca]
                       [--pca_method {sklearn,faiss}] [--pca_reduction PCA_REDUCTION]
                       [--pca_whitening] [--reassign_clustering] [--clustering {sklearn,faiss}]
                       [--metrics] [--metrics_file METRICS_FILE] [--metrics_dir METRICS_DIR]
                       [--checkpoint] [--checkpoint_file CHECKPOINT_FILE] [--verbose] [--seed SEED]
                       [--contrastive_strategy_1] [--contrastive_strategy_2] [--remove_head]
                       [--augmentation_resize AUGMENTATION_RESIZE]
                       [--augmentation_random_crop AUGMENTATION_RANDOM_CROP]
                       [--augmentation_random_rotation AUGMENTATION_RANDOM_ROTATION]
                       [--augmentation_random_horizontal_flip] [--augmentation_random_vertical_flip]

lored images)
                         - 1 for b/w images
  --num_classes NUM_CLASSES
                        The amount of classes are to be discovered and clustered by the CNN and k-Means algorithm. (default: 10)
  --sobel               Activates the Sobel filter for images. (default: False)
                        Note: Requires b/w image inputs, which can be obtained by also using the '--grayscale' flag.
  --grayscale           Reduces colored images to b/w images. (default: False)
  --requires_grad       Activates the requires_grad option for the input images in the training loop. Mainly used for analytical purposes (default: True)
  --epochs EPOCHS       Sets the training epochs for the model. (default: 100)
  --dataset {CIFAR10,MNIST,FashionMNIST,KMNIST,USPS,tinyimagenet,STL10,GTSRB}
                        Define which dataset a model is trained with. (default: MNIST)
  --data_dir DATA_DIR   Where the training data is locally downloaded and extracted. (default: /data)
  --ds_train            Selects the training images for certain datasets (default: False):
                         - MNIST
                         - CIFAR10
                         - FashionMNIST
                         - KMNIST

                        When not seltected, only the test images are downloaded, extracted and/or used.
  --ds_split {train,test,unlabeled,train+unlabeled,val}
                        Selects the type of data for sepcific datasets (default: train):
                         - tinyimagenet (train, val, test)
                         - STL10 (train, test, unlabeled, train+unlabeled)
                         - GTSRB (train, test)
  --batch_size BATCH_SIZE
                        Batch size for the main and training Dataset. (default: 256)
  --optimizer {SGD,Adam}
                        Main Optimizer for the complete Model. (default: SGD)
  --lr LR               Learning Rate for the main Optimizer. (default: 0.05)
  --momentum MOMENTUM   Momentum for the main Optimizer and only used for SGD Optimizer. (default 0.9)
  --weight_decay WEIGHT_DECAY
                        Weight Decay for the main Optimizer. (defualt: 10^-5)
  --beta1 BETA1         Beta1 value for the main Optimizer and only used for the Adam optimizer. (default: 0.9)
  --beta2 BETA2         Beta2 value for the main Optimizer and only used for the Adam optimizer. (default: 0.999)
  --param_requires_grad
  --reassign_optimizer_tl
                        If active, the optimizer for the top layer of the CNN will always be reset/reassigned for each epoch. (default: False)
  --optimizer_tl {SGD,Adam}
                        Top layer Optimizer for the complete Model. (default: SGD)
  --lr_tl LR_TL         Learning Rate for the top layer Optimizer. (default: 0.05)
  --momentum_tl MOMENTUM_TL
                        Momentum for the top layer Optimizer and only used for SGD Optimizer. (default 0.9)
  --weight_decay_tl WEIGHT_DECAY_TL
                        Weight Decay for the top layer Optimizer. (defualt: 10^-5)
  --beta1_tl BETA1_TL   Beta1 value for the top layer Optimizer and only used for the Adam optimizer. (default: 0.9)
  --beta2_tl BETA2_TL   Beta2 value for the top layer Optimizer and only used for the Adam optimizer. (default: 0.999)
  --loss_fn {L1,L2,MSE,CrossEntropy}

                        The preferred PCA implementation. (default: faiss)
  --pca_reduction PCA_REDUCTION
                        Up to how many components the features are reduced. (default: 256)
  --pca_whitening       When active, the selected PCA reduction method will also perform whitening of the dataset. (default: False)
  --reassign_clustering
                        When active, the selected clustering method will always reassigned before a new clustering is executed. (default: False)
  --clustering {sklearn,faiss}
                        Which clustering implementation of k-Means DeepCluster is using. (default: faiss)
  --metrics             When active, metrics regarding the DeepCluster model are printed and stored in a dedicated metrics folder. (default: False)
  --metrics_file METRICS_FILE
                        Define a specific metrics file path when resuming a previous training. This is requires also to set a checkpoint file, otherwise the algorithm will start from the beginning and simply add data starting at the first epoch. (default: None)
  --metrics_dir METRICS_DIR
                        Define a specific metrics storage directory when running specific tests. (default: None)
  --checkpoint          When active, checkpoints are continiously created at each epoch and additionally a best model checkpoint. (default: False)
  --checkpoint_file CHECKPOINT_FILE
                        Define a file path for a checkpoint when the intention is to resume a previous run model. (default: None)
  --verbose, -v         Print further information when running the model. (default: None)
  --seed SEED           Define a seed that is used when initializing the model. (default: None)
  --contrastive_strategy_1
                        When active, the first contrastive strategy is used. The image is feed into the DeepCluster framework. Together with the resulting pseudo labels the features are fed into a contrastive head, that calculates the contrastive loss. The sum of the DeepCluster Loss and the Contrastive Loss will be the final loss. (default: False)
  --contrastive_strategy_2
                        When active, the second contrastive strategy is used. The input image is augmented into two different verions of the origin. Both augme nted images are feed into the DeepCluster framework. both representations are used to calculate the NTXentLoss. The final loss will be the sum of both DeepCluster losses and the NTXent Loss. (default: False)
  --remove_head         When active, the top layer head (final classifier) will be rmeoved and later reattached, as it is done by the original implementation.
  --augmentation_resize AUGMENTATION_RESIZE
                        Resize the input images to the given size.
  --augmentation_random_crop AUGMENTATION_RANDOM_CROP
                        Randomly crop the input images to the given size.
  --augmentation_random_rotation AUGMENTATION_RANDOM_ROTATION
                        Randomly rotate the input images to a random degree. between the given integer and the negative of the given integer.
  --augmentation_random_horizontal_flip
                        Randomly flip the input images horizontally.
  --augmentation_random_vertical_flip
                        Randomly flip the input images vertically.
  --augmentation_color_jitter
                        Randomly change the brightness, contrast and saturation of the input images.
  --augmentation_random_autocontrast
                        Randomly apply autocontrast to the input images.
  --augmentation_random_equalize
                        Randomly apply equalize to the input images.
```
