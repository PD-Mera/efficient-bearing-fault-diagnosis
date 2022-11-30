import nni

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import nni.retiarii.evaluator.pytorch.lightning as pl

from data import *
from config import *


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (_, data, temp, target) in enumerate(train_loader):
        data, temp, target = data.to(device), temp.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, temp)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data, temp, target in test_loader:
            data, temp, target = data.to(device), temp.to(device), target.to(device)
            output = model(data, temp)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(str(correct) + '/' + str(len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_data = LoadDataset(phase='train')
    test_data  = LoadDataset(phase='test')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    # test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    for epoch in range(TRAINING_EPOCH):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        torch.save(model, 'model.pth')
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)

