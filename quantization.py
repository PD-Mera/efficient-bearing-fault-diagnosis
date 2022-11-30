import torch
from torch.optim import SGD
from torchvision import transforms
from nni.compression.pytorch.quantization import QAT_Quantizer
from PIL import Image
import nni
import argparse

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import nni.retiarii.evaluator.pytorch.lightning as pl
# from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

from data import *
from config import *
from model import Net
    

    


config_list = [{
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_types': ['Conv2d', 'Conv1d', 'Linear']
}, {
    'quant_types': ['output'],
    'quant_bits': {'output': 8},
    'op_types': ['BatchNorm2d']
}, 
# {
#     'quant_types': ['input', 'weight'],
#     'quant_bits': {'input': 8, 'weight': 8},
#     'op_names': ['fc1', 'fc2']
# }
]

def rpf(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


def create_empty_square_matrix(dim):
    return torch.zeros((5, 5))

def add_to_confusion_matrix(matrix, position_h, position_w):
    matrix[position_h][position_w] += 1
    return matrix


    

def load_img_to_tensor(img_link):
    img = Image.open(img_link)
    tensor_img = transforms.ToTensor()(img)
    return tensor_img



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


def test_epoch(model, device, test_loader, pos):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = create_empty_square_matrix(5)

    with torch.no_grad():
        for _, data, temp, target in test_loader:
            data, temp, target = data.to(device), temp.to(device), target.to(device)
            output = model(data, temp)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            for pos_h, pos_w in zip(target.squeeze(1), pred.squeeze(1)):
                confusion_matrix = add_to_confusion_matrix(confusion_matrix, pos_h, pos_w)

            correct += pred.eq(target.view_as(pred)).sum().item()

    torch.save(confusion_matrix, f'confusion_matrix/cm_{pos}.pth')

    test_loss /= len(test_loader.dataset)
    print(str(correct) + '/' + str(len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls, optimizer, pos):
    # "model_cls" is a class, need to instantiate
    model = model_cls

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_data = LoadDataset(phase='train')
    test_data  = LoadDataset(phase='test')
    
    # transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    # test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    for epoch in range(TRAINING_EPOCH):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader, pos)
        torch.save(model.state_dict(), 'model.pth')
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', default=1, help='Number of training')

    args = parser.parse_args()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Net()
    
    # model.load_state_dict(torch.load('model_7_5_original.pth'))
    model = model.to(device)
    optimizer = SGD(model.parameters(), 1e-2)
    dummy_input_1 = torch.rand(32, 1, 16, 16).to(device)
    dummy_input_2 = torch.rand(32, 1, 256).to(device)
    quantizer = QAT_Quantizer(model, config_list, optimizer, (dummy_input_1, dummy_input_2))
    quantizer.compress()
    print(model)
    

    evaluate_model(model, optimizer, args.pos)
    model_path = "model_7_5.pth"
    calibration_path = f"calib/calib_config_{args.pos}.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    print(calibration_config)

    

    
    # input_shape = (32, 1, 28, 28)
    # engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
    # engine.compress()
    # test_trt(engine)

    # img = load_img_to_tensor('/root/workspace_2/Bearing-Dataset-16x16-noise/train/I/I001.png')
    # print(img.size())