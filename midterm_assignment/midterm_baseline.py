# Import Libraries
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Configure
args = {}
kwargs = {}
args['batch_size'] = 2
args['test_batch_size'] = 1000
args['epochs'] = 10  # The number of Epochs is the number of times you go through the full dataset.
args['lr'] = 0.01  # Learning rate is how fast it will decend.
# SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).
args['momentum'] = 0.5  # SGD momentum (default: 0.5) Momentum is a moving average of our gradients
#                        # (helps to keep direction).

args['seed'] = 1  # random seed
args['log_interval'] = 5000 // args['batch_size']
args['cuda'] = True
args['norm_type'] = 'batch'

# Load the data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv1_gn = nn.GroupNorm(2, 10)
        self.conv1_norm = self.conv1_bn if args['norm_type'] == 'batch' else self.conv1_gn

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # Dropout
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_gn = nn.GroupNorm(4, 20)
        self.conv2_norm = self.conv2_bn if args['norm_type'] == 'batch' else self.conv2_gn

        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc1_gn = nn.GroupNorm(10, 50)
        self.fc1_norm = self.fc1_bn if args['norm_type'] == 'batch' else self.fc1_gn

        self.fc2 = nn.Linear(50, 10)

    def LeakyReLU(self, x):
        return (x > 0.1 * x).float() * x + (x < 0.1 * x).float() * 0.1 * x

    def forward(self, x):
        # Convolutional Layer/Pooling Layer/Activation
        x = self.LeakyReLU(self.conv1_norm(F.max_pool2d(self.conv1(x), 2)))
        # Convolutional Layer/Dropout/Pooling Layer/Activation
        x = self.LeakyReLU(self.conv2_norm(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, 320)
        # Fully Connected Layer/Activation
        x = self.LeakyReLU(self.fc1_norm(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        # Fully Connected Layer/Activation
        x = self.fc2(x)
        # Softmax gets probabilities.
        return F.log_softmax(x, dim=1)


model = Net()
if args['cuda']:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        # Variables in Pytorch are differentiable.
        data, target = Variable(data), Variable(target)
        # This will zero out the gradients for this batch.
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss.
        # It is useful to train a classification problem with C classes.
        loss = F.nll_loss(output, target)
        # dloss/dx for every Variable
        loss.backward()
        # to do a one-step update on our parameter.
        optimizer.step()
        # Print out the loss periodically.
        # if batch_idx % args['log_interval'] == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.data))


final_accuracy = {'batch': [], 'group': []}


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    if epoch == args['epochs']:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))



for epoch in range(1, args['epochs'] + 1):
    train(epoch)
    test(epoch)

print(final_accuracy)
