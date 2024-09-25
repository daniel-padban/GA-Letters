import torch
import torch.nn as nn


class CNN(nn.Module):
    '''
    Takes input of [channels, height, widthI]
    height = width
    padding = 0
    stride (convolution) = 1
    stride (pooling) = pooling kernel

    :param conv*: Output channels of convolutional layer *
    '''
    def __init__(self,
                input_channels:int,
                conv1:int,
                ckernel1:int,
                MPkernel1:int,
                conv2:int,
                ckernel2:int,
                MPkernel2:int,

                out_dim:int=26,
                input_HW:int = 28):
        super().__init__()
        
        #Convolution-Pooling block 1:
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=conv1,kernel_size=ckernel1,padding='same')
        self.le_relu1 = nn.LeakyReLU()
        self.mpool1 = nn.MaxPool2d(kernel_size=MPkernel1)
        mpool1_HW = self.calculate_dims(input_HW,MPkernel1,stride=MPkernel1) #height and/or width of mpool1 output
        self.bn1 = nn.BatchNorm2d(num_features=conv1)

        #Convolution-Pooling block 2:
        self.conv2 = nn.Conv2d(in_channels=conv1,out_channels=conv2,kernel_size=ckernel2,padding='same')
        self.le_relu2 = nn.LeakyReLU()

        self.mpool2 = nn.MaxPool2d(kernel_size=MPkernel2)
        mpool2_HW = self.calculate_dims(mpool1_HW,MPkernel2,stride=MPkernel2) #height and/or width of mpool2 output


        #output block
        self.flatten = nn.Flatten()
        flattened_dim = int((mpool2_HW**2)*conv2)
        self.fc1 = nn.Linear(in_features=flattened_dim,out_features=128)
        self.le_relu3 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout()
        self.fco = nn.Linear(in_features=128,out_features=out_dim)

    def calculate_dims(self, HW:int, kernel:int,stride:int=1,):
        '''
        :param HW: height x width, h = w
        :param kernel: kernel size
        '''
        return (((HW-kernel + 2*0)/stride)+1)

    def forward(self, x):
        #Convolution-Pooling block 1:
        x = self.conv1(x)
        x = self.le_relu1(x)
        x = self.mpool1(x)
        x = self.bn1(x)
        #Convolution-Pooling block 2:
        x = self.conv2(x)
        x = self.le_relu2(x)
        x = self.mpool2(x)

        #Outputs
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.le_relu3(x)
        #x = self.dropout1(x)
        output = self.fco(x)
        return output

