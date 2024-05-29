import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # first layer
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.relu1 = nn.ReLU()

        # first max pool layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # second layer
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.relu2 = nn.ReLU()

        # second max pool layer
        self.pool2 = nn.MaxPool2d(2, 2)

        # first fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()

        # second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        # output layer
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a subset of the training dataset with 1000 samples
    subset_indices = np.random.choice(len(full_train_dataset), 1000, replace=False)
    train_subset = Subset(full_train_dataset, subset_indices)

    # Create DataLoaders for the subset and full test dataset
    train_dl = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2)
    test_dl = DataLoader(full_test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Verify the size of the training subset
    print(f'Training subset size: {len(train_subset)}')
    print(f'Test set size: {len(full_test_dataset)}')


    # initializing parameters
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 10
    batch_size = 16

    # training the net
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # training the model
        for step, data in enumerate(tqdm(train_dl, desc='batch')):
            input, labels = data

            optimizer.zero_grad()

            output = net(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            if step % 2000 == 1999:  # print every 2000 mini-batches
                tqdm.write(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)
        print(f'Accuracy of the network on the training set after epoch {epoch + 1}: {train_accuracy:.2f} %')

        # evaluating the model on test data
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in test_dl:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

    print('Finished Training')
