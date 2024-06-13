import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import math
import copy
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from models.VGG import VGG16
from models.AlexNet import AlexNet

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

import faiss

class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument. This assigns
    each image with its "pseudolabel"
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        img, pseudolabel = self.imgs[index]
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr

class SimpleCnn(nn.Module):
    
    def __init__(self, num_channels, k=10, input_dim=28):
        
        super(SimpleCnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.classifier = nn.Sequential(
            # nn.Linear(7*7*16, 64),
            nn.Linear(input_dim**2, 64),
            nn.ReLU()
        )

        self.top_layer = nn.Linear(64, k)
        self._initialize_weights()
    
    def forward(self, x):
        
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        if self.top_layer:
            out = self.top_layer(out)
        return out
    
    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def train_supervised(model, device, train_loader, epoch):
    model.train()
    torch.set_grad_enabled(True)
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**(-5)
    )

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    for e in range(epoch):

      epoch_loss = 0.0

      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          epoch_loss += output.shape[0] * loss.item()

      print("Epoch Nr: " + str(e))
      print(epoch_loss / len(train_loader.dataset))
      
def calculate_accuracy(true_labels, predicted_labels):
    # Create a contingency table
    contingency_table = np.zeros((max(true_labels) + 1, max(predicted_labels) + 1), dtype=int)
    for true_label, pred_label in zip(true_labels, predicted_labels):
        contingency_table[true_label, pred_label] += 1

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-contingency_table)

    # Create a new array with reassigned predicted labels
    new_predicted_labels = np.zeros_like(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        new_predicted_labels[predicted_labels == j] = i

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, new_predicted_labels)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    
    test_loss = 0
    predicted_labels = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            predicted_labels.append(pred)
            true_labels.append(target)

    test_loss /= len(test_loader.dataset)
    
     # calculate NMI
    predicted_labels = torch.cat(predicted_labels).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()
    
    #ValueError: labels_pred must be 1D: shape is (10000, 1)
    predicted_labels = np.squeeze(predicted_labels)
    true_labels = np.squeeze(true_labels)
    
    print("\n", "="*25, "Test Results", "="*25)
    
    print("Average Loss: " + str(test_loss))
    print("NMI: " + str(normalized_mutual_info_score(true_labels, predicted_labels)))
    print("Calculated Accuracy: " + str(calculate_accuracy(true_labels, predicted_labels)))
    
def compute_features(dataloader, model, N, get_labels=False):

    model.eval()
    labels = []

    # discard the label information in the dataloader
    for i, (input_tensor, label) in enumerate(dataloader):
        # print("Input tensor shape", input_tensor.shape)
        input_var = torch.autograd.Variable(input_tensor.cuda(), requires_grad=False)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * 64: (i + 1) * 64] = aux
        else:
            # special treatment for final batch
            features[i * 64:] = aux

        # measure elapsed time

        labels.append(label.numpy())

    labels = np.concatenate(labels)

    if get_labels:
      return features, labels
    
    else:
      return features
  
def cluster_assign(images_lists, dataset, transformation):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    return ReassignedDataset(image_indexes, pseudolabels, dataset, transformation)

def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    losses = AverageMeter()
    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.01,
        weight_decay=10**-5,
    )

    for i, (input_tensor, target) in enumerate(loader):

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

    return losses.avg
     
def DeepCluster(model, device, train_loader, epoch, k, transformation):

    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None

    model = model.to(device)


    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**(-5)
    )

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    #cluster_step


    for e in range(epoch):
         
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        features, true_labels = compute_features(train_loader, model, len(unsupervised_pretrain), get_labels=True)
      
      # do PCA reduction to 256 dimensions if the original feature is higher-dimensional
        if features.shape[1] > 256:
            features = PCA(n_components=256).fit_transform(features)
            # normalize the feature
            features = normalize(features, norm='l2')
    
        dims = features.shape[1]

        # faiss implementation of k-means
        clus = faiss.Clustering(dims, k)
        clus.seed = np.random.randint(1234)

        clus.niter = 20
        clus.max_points_per_centroid = 60000

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, dims, flat_config)

        #get new cluster labels
        clus.train(features, index)
        _, I = index.search(features, 1)

        labels = np.squeeze(I)

        unique, counts = np.unique(labels, return_counts=True)

        images_lists = [[] for i in range(k)]
        for i in range(len(unsupervised_pretrain)):
            images_lists[int(labels[i])].append(i)


        # create new dataset from pseudolabels
        train_dataset = cluster_assign(images_lists, unsupervised_pretrain, transformation)

        #print(len(train_dataset))
        #print(images_lists)

        # sample images from uniform distribution over classes
        sampler = UnifLabelSampler(int(1 * len(train_dataset)),
                                    images_lists)


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=4,
            sampler=sampler,
        )
      
        # reset last layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, k)
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train step
        torch.set_grad_enabled(True)
        loss = train(train_dataloader, model, criterion, optimizer, e)
        
        print("=" * 25, "Epoch Nr: " + str(e), "=" * 25)

        print("Epoch Loss:", loss.cpu().numpy())
        print("Overview of cluster assignments:")
        print(dict(zip(unique, counts)))
        print("NMI score:", normalized_mutual_info_score(true_labels, labels))
        print("Accuracy score:", calculate_accuracy(true_labels, labels))

if __name__ == "__main__":
    # seed everything
    torch.manual_seed(42)

    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform=transforms.Compose([
            # resize to 32x32 if mnist
            # transforms.Resize((32, 32)),
            # for alexnet we have to resize to 224x224
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Magic numbers for MNIST
            # transforms.Normalize((0.1307,), (0.3081,))
            # Magic numbers for CIFAR10
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # mnist_test = datasets.MNIST('../data', train=False, transform=transform)
    
    cifar_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10('../data', train=False, transform=transform)

    # unsupervised_pretrain = mnist_train
    unsupervised_pretrain = cifar_train
    train_loader_unsupervised = torch.utils.data.DataLoader(unsupervised_pretrain, batch_size=64, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=True, num_workers=4)
    
    # simpleCNN = SimpleCnn(
    #     num_channels=train_loader_unsupervised.dataset[0][0].shape[0],
    #     k=10,
    #     input_dim=train_loader_unsupervised.dataset[0][0].shape[1]
    # )
    
    # simpleCNN = simpleCNN.to(device)
    
    # DeepCluster(
    #     model=simpleCNN, 
    #     device=device,
    #     train_loader=train_loader_unsupervised,
    #     epoch=1,
    #     k=10,
    #     transformation=transform
    # )
    
    # vgg16 = VGG16(
    #     input_dim=train_loader_unsupervised.dataset[0][0].shape[0],
    #     num_classes=10, 
    #     sobel=False,
    #     input_size=train_loader_unsupervised.dataset[0][0].shape[1]
    # )
    # vgg16 = vgg16.to(device)
    # DeepCluster(
    #     model=vgg16,
    #     device=device,
    #     train_loader=train_loader_unsupervised,
    #     epoch=5,
    #     k=10,
    #     transformation=transform
    # )
    
    alexnet = AlexNet(
        input_dim=train_loader_unsupervised.dataset[0][0].shape[0],
        num_classes=10,
        sobel=False,
        input_size=train_loader_unsupervised.dataset[0][0].shape[1]
    )
    alexnet = alexnet.to(device)
    DeepCluster(
        model=alexnet,
        device=device,
        train_loader=train_loader_unsupervised,
        epoch=5,
        k=10,
        transformation=transform
    )
    
    # linear_model(simpleCNN, train_loader_supervised, test_loader)
    # linear_model(vgg15, train_loader_supervised, test_loader)
    
    # Check accuracy on test set
    # test(simpleCNN, device, test_loader)
    # test(vgg16, device, test_loader)
    test(alexnet, device, test_loader)
    
    print("DONE!")

