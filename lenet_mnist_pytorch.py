# import the necessary packages
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.fc1 = nn.Linear(in_features=4*4*50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(args, model, device, optimizer, train_loader, epoch):
    '''
    Network training function
    ---
    Args:
        args: parsed arguments
        model: CNN network architecture
        device: gpu or cpu
        optimizer: algorithm used for weight and bias update
        train_loader: data loader for training set
        epoch: number of complete passes (forward pass and backpropagation)
                    by the training set through the network

    Returns:
        training loss
    '''
    # state that we are training the model
    model.train()

    # loss function
    loss_func = nn.CrossEntropyLoss()

    # iterate over batches of data per epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # wrap the input and target output in the 'Variable' wrapper
        data, target = data.to(device), target.to(device)

        # clear the gradients as PyTorch accumulates them
        optimizer.zero_grad()

        # forward pass
        output = model(data)
        train_loss = loss_func(output, target)

        # backpropagation
        train_loss.backward()

        # update the parameters (weight, bias)
        optimizer.step()

        # print training log
        if batch_idx % args.log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch,
                         batch_idx*len(data),
                         len(train_loader.dataset),
                         100.*batch_idx/len(train_loader),
                         train_loss.data))


def test(args, model, device, test_loader, epoch):
    '''
    Network testing function
    ---
    Args:
        args: parsed arguments
        model: CNN network architecture
        device: gpu or cpu
        test_loader: data loader for testing set
        epoch: number of complete passes (forward pass and backpropagation)
                    by the training set through the network

    Returns:
        testing loss
    '''
    # state that we are testing the model; this prevents layers
    # such as Dropout to take effect
    model.eval()

    # init loss & correct prediction accumulators
    test_loss = 0
    correct = 0

    # loss function
    loss_func = nn.CrossEntropyLoss(size_average=False)

    # iterate over batches of data per epoch
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # forward pass
        output = model(data)
        test_loss += loss_func(output, target).data

        # retrieve the index of the max log-probability
        # (the predicted output label)
        pred = output.argmax(dim=1, keepdim=True)

        # if label is predicted correctly, we increase the
        # correct prediction accumulator
        correct += pred.eq(target.view_as(pred)).sum().item()

    # print testing log
    test_loss /= len(test_loader.dataset)

    print('\nTest set, Epoch {}, Average loss: {: .4f}, Accuracy: {}/{}\
          ({: .0f} % )\n'.format(epoch,
                                 test_loss,
                                 correct,
                                 len(test_loader.dataset),
                                 100.*correct /
                                 len(test_loader.dataset)))


def main():
    '''
    Training settings
    '''
    parser = argparse.ArgumentParser(
        description='PyTorch MNIST Example')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        metavar='M',
        help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before training log status ')
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=False,
        help='for saving the current model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # training data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.3081, ))
                       ])),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.3081, ))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # train and evaluate network
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, optimizer, train_loader, epoch)
        test(args, model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(), 'lenet_mnist.pt')


if __name__ == '__main__':
    main()
