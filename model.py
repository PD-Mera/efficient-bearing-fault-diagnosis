import torch
import torch.nn.functional as F
# import torch.nn as nn

import nni.retiarii.nn.pytorch as nn
import nni
from nni.retiarii import model_wrapper

from config import *



class Net(nn.Module):
    def __init__(self, input_size = 16, input_channel = 2, num_class = 5):
        super().__init__()
        # kernel_size = nn.ValueChoice([3, 5, 7])
        # feature = nn.ValueChoice([5, 10, 15, 20])
        kernel_size = 7
        feature = 5
        # input_size = 16
        output_size = input_size - kernel_size + 3


        self.conv = nn.Conv2d(input_channel, feature, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(feature)
        self.fc = nn.Linear(output_size * output_size * feature, num_class)

        self.conv_temp = nn.Conv1d(
            in_channels = 1, 
            out_channels = 16, 
            kernel_size = 31, 
            stride = 15, 
            padding = 0)

    def forward(self, x, y):
        y = self.conv_temp(y).unsqueeze(1)
        x = torch.concat((x, y), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output



@model_wrapper      # this decorator should be put on the out most
class BaseModelSpace(nn.Module):
    def __init__(self, input_size = 16, input_channel = 2, num_class = 5):
        super().__init__()
        kernel_size_step_1 = nn.ValueChoice([3, 5, 7])
        feature_step_1 = nn.ValueChoice([5, 10, 15, 20])
        # input_size = 16
        output_size = input_size - kernel_size_step_1 + 3

        # image
        self.conv = nn.Conv2d(
            in_channels = input_channel, 
            out_channels = feature_step_1, 
            kernel_size = kernel_size_step_1, 
            stride = 1, 
            padding = 1)
        self.bn = nn.BatchNorm2d(feature_step_1)
        self.fc = nn.Linear(output_size * output_size * feature_step_1, num_class)

        self.conv_temp = nn.Conv1d(
            in_channels = 1, 
            out_channels = 16, 
            kernel_size = 31, 
            stride = 15, 
            padding = 0)


    def forward(self, x, y):
        y = self.conv_temp(y).unsqueeze(1)
        x = torch.concat((x, y), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output
