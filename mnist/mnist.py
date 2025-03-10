from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from torch.nn import MaxPool2d
import time
from prettytable import PrettyTable



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

train_loader_rotate = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,),
                       transforms.RandomApply([transforms.RandomRotation((45, 45))], p=0.5),
)
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class GroupNet(nn.Module):
    def __init__(self):
        super(GroupNet, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(4*4*20*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        # print('GroupNet: fc1 x shape = ', x.shape) # 64x1280
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.conv3 = nn.Conv2d(10, 20, 3)
        self.conv4 = nn.Conv2d(20, 20, 3)

        # self.fc1 = nn.Linear(4*4*20*4, 50)
        self.fc1 = nn.Linear(4*4*20, 50) # reduced parameters in the fully convolutional layers.
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)

        #untimeError: mat1 and mat2 shapes cannot be multiplied (64x320 and 1280x50)
        # print('Net: fc1 x shape = ', x.shape) # 64x320

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10 * 2, 3)
        self.conv2 = nn.Conv2d(10 * 2, 10 * 2, 3)
        self.conv3 = nn.Conv2d(10 * 2, 25, 3)
        self.conv4 = nn.Conv2d(25, 20 * 4, 3)
        self.fc1 = nn.Linear(4*4*20*4, 50) 
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)

        #untimeError: mat1 and mat2 shapes cannot be multiplied (64x320 and 1280x50)
        # print('Net: fc1 x shape = ', x.shape) # 64x320

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



def train(model, optimizer, epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='\r')

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    acc_percent = 100.0 * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return acc_percent.item()


def plot_metrics(NetClass, log_file, epochs, train_loader):
    model = NetClass()
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    start = time.time()
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader)
        acc_percent = test(model)
        print(f'{epoch},{round(acc_percent, 4)},{start},{time.time()}', file=log_file)


def plot_all_metrics():
    epochs = 20

    for i in range(10):
        train_log = open('logs/cnn_big_no_group_' + str(i).zfill(2) + '.csv', 'w+')
        print('epoch,test_accuracy,start_time,cur_time', file=train_log)
        plot_metrics(BigNet, train_log, epochs, train_loader)

    for i in range(10):
        train_log = open('logs/cnn_no_group_' + str(i).zfill(2) + '.csv', 'w+')
        print('epoch,test_accuracy,start_time,cur_time', file=train_log)
        plot_metrics(Net, train_log, epochs, train_loader)

    for i in range(10):
        train_log = open('logs/cnn_group_' + str(i).zfill(2) + '.csv', 'w+')
        print('epoch,test_accuracy,start_time,cur_time', file=train_log)
        plot_metrics(GroupNet, train_log, epochs, train_loader)

    for i in range(10):
        train_log = open('logs/cnn_big_no_group_aug' + str(i).zfill(2) + '.csv', 'w+')
        print('epoch,test_accuracy,start_time,cur_time', file=train_log)
        plot_metrics(BigNet, train_log, epochs, train_loader_rotate)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def count_params_for_all_models():
    print('Big Net params:', count_parameters(BigNet()))
    print('GroupNet params:', count_parameters(GroupNet()))


if __name__ == '__main__':

    # count_params_for_all_models()
    plot_all_metrics()
