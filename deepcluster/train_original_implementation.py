import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler
import torchvision
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

from models.VGG import VGG16
from models.AlexNet import AlexNet


# taken and refactored from 'https://github.com/facebookresearch/deepcluster/blob/main/clustering.py'
def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ReassignedDataset(torch.utils.data.Dataset):
    """A dataset where the new images labels are given in argument.
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
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

def cluster_assign(images_lists, dataset):
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

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    
    # normalization for MNIST
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    _, d = x.shape

    # Faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change Faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # Perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    # If `clus.obj` does not exist, manually track the loss
    def track_loss(clus):
        niter = clus.niter
        d = clus.d
        k = clus.k
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
        loss = []
        for i in range(niter):
            clus.niter = i + 1
            clus.train(x, index)
            _, I = index.search(x, 1)
            current_loss = np.mean([np.linalg.norm(x[j] - centroids[I[j][0]])**2 for j in range(len(x))])
            loss.append(current_loss)
        return np.array(loss)

    losses = track_loss(clus)
    print(f'k-means loss evolution: {losses}')

    return [int(n[0]) for n in I], losses[-1]

def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
    
def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]

# taken and refactored from 'https://github.com/facebookresearch/deepcluster/blob/main/util.py'
class UnifLabelSampler(Sampler):
    """
    Samples elements uniformly across pseudolabels.

    Args:
        N (int): Size of the returned iterator.
        images_lists (dict): Dictionary where the key is the target and the value is a list of data with this target.
    """

    def __init__(self, n, images_lists):
        super().__init__(Sampler)
        self.n = n
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        # check if self.images_lists are numpy arrays
        
        print("this is the type of the first element", type(self.images_lists[0]))
        
        # print("this is the list", self.images_lists)
        # print("this is the type of the first element", type(self.images_lists[0][0]))
        
        non_empty_clusters = [lst for lst in self.images_lists if lst]
        nmb_non_empty_clusters = len(non_empty_clusters)
        size_per_pseudolabel = int(self.n / nmb_non_empty_clusters) + 1
        
        res = []

        for cluster in non_empty_clusters:
            indexes = np.random.choice(
                cluster,
                size_per_pseudolabel,
                replace=(len(cluster) <= size_per_pseudolabel)
            )
            res.extend(indexes)
        
        np.random.shuffle(res)
        res = res[:self.n] if len(res) >= self.n else res + res[:self.n - len(res)]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
    
class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

# taken from 'https://github.com/facebookresearch/deepcluster/blob/main/main.py'
def compute_features(dataloader, model, N, verbose, batch):
    if verbose:
        print('Compute features')
    # batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch: (i + 1) * batch] = aux
        else:
            # special treatment for final batch
            features[i * batch:] = aux

        # measure elapsed time
        # batch_time.update(time.time() - end)
        end = time.time()

        # if verbose and (i % 200) == 0:
        #     print('{0} / {1}\t'
        #           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
        #           .format(i, len(dataloader), batch_time=batch_time))
    return features

def train(loader, model, crit, opt, epoch, lr, wd, verbose ):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # data_time = AverageMeter()
    # forward_time = AverageMeter()
    # backward_time = AverageMeter()
    
    losses = []

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=lr,
        weight_decay=10**wd,
    )

    # end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        # data_time.update(time.time() - end)

        # save checkpoint
        # n = len(loader) * epoch + i
        # if n % checkpoints == 0:
        #     path = os.path.join(
        #         exp,
        #         'checkpoints',
        #         'checkpoint_' + str(n / checkpoints) + '.pth.tar',
        #     )
        #     if verbose:
        #         print('Save checkpoint at: {0}'.format(path))
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'arch': arch,
        #         'state_dict': model.state_dict(),
        #         'optimizer' : opt.state_dict()
        #     }, path)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        # losses.update(loss.data[0], input_tensor.size(0))
        losses.append(loss.item())

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        end = time.time()

        # if verbose and (i % 200) == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss: {loss.val:.4f} ({loss.avg:.4f})'
        #           .format(epoch, i, len(loader), batch_time=batch_time,
        #                   data_time=data_time, loss=losses))
        
    return torch.tensor(losses).mean().item()

def main():
    # fix random seeds
    seed = 42
    lr = 0.05
    wd = -5
    momentum = 0.9
    batch = 32
    workers = 4
    epochs = 50
    reassign = 1
    exp = 'exp'
    arch = 'alexnet'
    nmb_cluster = 10 # 10 for CIFAR-10 and MNIST
    
    verbose = True
    
    print("Settinig seeds")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Seeds set")

    if arch == 'alexnet':
        model = AlexNet(input_dim=3, num_classes=10, sobel=False)
    elif arch == 'vgg16':
        model = VGG16(input_dim=3, num_classes=10, sobel=False)
    else:
        raise NotImplementedError
    
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=10**wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    # if resume:
    #     if os.path.isfile(resume):
    #         print("=> loading checkpoint '{}'".format(resume))
    #         checkpoint = torch.load(resume)
    #         start_epoch = checkpoint['epoch']
    #         # remove top_layer parameters from checkpoint
    #         for key in checkpoint['state_dict']:
    #             if 'top_layer' in key:
    #                 del checkpoint['state_dict'][key]
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(resume))

    # creating checkpoint repo
    # exp_check = os.path.join(exp, 'checkpoints')
    # if not os.path.isdir(exp_check):
    #     os.makedirs(exp_check)
    
    if not os.path.isdir(exp):
        os.makedirs(exp)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(exp, 'clusters'))
    
    # normalization for MNIST
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    # preprocessing of data
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    print("Loading data")
    # pt_dataset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.Compose(tra))
    
    # for idx, (img, label) in tqdm(enumerate(pt_dataset), desc='Saving images', total=len(pt_dataset)):
    #     # check if directory exists
    #     if not os.path.exists(f"data/MNIST/images/{label}"):
    #         os.makedirs(f"data/MNIST/images/{label}")
        
    #     # check if image exists
    #     if os.path.exists(f"data/MNIST/images/{label}/{idx}.jpg"):
    #         continue
        
    #     img = transforms.ToPILImage()(img)
    #     img.save(f"data/MNIST/images/{label}/{idx}.jpg")
    
    dataset = datasets.ImageFolder("data/MNIST/images", transform=transforms.Compose(tra))
    print("Data loaded")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch,
                                             num_workers=workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = Kmeans(nmb_cluster)
    
    nmi_against_true_labels = []
    nmi_against_previous_assignment = []
    all_losses = []

    # training convnet with DeepCluster
    for epoch in tqdm(range(epochs), desc='Epochs', total=epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset), verbose, batch)

        # cluster the features
        if verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=verbose)

        # assign pseudo-labels
        if verbose:
            print('Assign pseudo labels')
        train_dataset = cluster_assign(deepcluster.images_lists, dataset.imgs)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch,
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch, lr, wd, verbose)
        all_losses.append(loss)

        # print log
        if verbose:
            print(f'###### Epoch [{epoch}] ###### \n'
                    f'Time: {time.time() - end:.3f} s\n'
                    f'Clustering loss: {clustering_loss:.3f} \n'
                    f'ConvNet loss: {loss:.3f}')
            
            # calculate NMI against true labels
            true_labels = np.array([dataset.imgs[idx][1] for idx in range(len(dataset))])
            nmi = normalized_mutual_info_score(
                true_labels,
                arrange_clustering(deepcluster.images_lists)
            )
            print(f'NMI against true labels: {nmi}')
            
            # save NMI against previous assignment
            nmi_against_true_labels.append(nmi)

            try:
                nmi = normalized_mutual_info_score(
                    arrange_clustering(deepcluster.images_lists),
                    arrange_clustering(cluster_log.data[-1])
                )
                print(f'NMI against previous assignment: {nmi}')
                
                # save NMI against previous assignment
                nmi_against_previous_assignment.append(nmi)
                
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        # torch.save({'epoch': epoch + 1,
        #             'arch': arch,
        #             'state_dict': model.state_dict(),
        #             'optimizer' : optimizer.state_dict()},
        #            os.path.join(exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)
        
    # save NMI against true labels
    with open(os.path.join(exp, 'nmi_against_true_labels.pkl'), 'wb') as f:
        pickle.dump(nmi_against_true_labels, f)
        
    # save NMI against previous assignment
    with open(os.path.join(exp, 'nmi_against_previous_assignment.pkl'), 'wb') as f:
        pickle.dump(nmi_against_previous_assignment, f)
        
    # save all losses
    with open(os.path.join(exp, 'all_losses.pkl'), 'wb') as f:
        pickle.dump(all_losses, f)
        
if __name__ == '__main__':
    main()