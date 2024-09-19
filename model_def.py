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
                conv3:int,
                ckernel3:int,
                MPkernel3:int,
                out_dim:int=26,
                input_HW:int = 28):
        super().__init__()
        
        #Convolution-Pooling block 1:
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=conv1,kernel_size=ckernel1)
        self.mpool1 = nn.MaxPool2d(kernel_size=MPkernel1)
        conv1_HW = self.calculate_dims(input_HW,ckernel1) #height and/or width of conv2 output
        mpool1_HW = self.calculate_dims(conv1_HW,MPkernel1,stride=MPkernel1) #height and/or width of mpool1 output

        #Convolution-Pooling block 2:
        self.conv2 = nn.Conv2d(in_channels=conv1,out_channels=conv2,kernel_size=ckernel2)
        self.mpool2 = nn.MaxPool2d(kernel_size=MPkernel2)
        conv2_HW = self.calculate_dims(mpool1_HW,ckernel2) #height and/or width of conv2 output
        mpool2_HW = self.calculate_dims(conv2_HW,MPkernel2,stride=MPkernel2) #height and/or width of mpool2 output

        #Convolution-Pooling block 3:
        self.conv3 = nn.Conv2d(in_channels=conv2,out_channels=conv3,kernel_size=ckernel3)
        self.mpool3 = nn.MaxPool2d(kernel_size=MPkernel3)
        conv3_HW = self.calculate_dims(mpool2_HW,ckernel3) #height and/or width of conv1 output
        mpool3_HW = self.calculate_dims(conv3_HW,MPkernel3,stride=MPkernel3) #height and/or width of mpool3 output


        #output block
        self.flatten = nn.Flatten()
        flattened_dim = (mpool3_HW**2)*conv3
        self.fco = nn.Linear(in_features=flattened_dim,out_features=out_dim)

    def calculate_dims(self, HW:int, kernel:int,stride:int=1,):
        '''
        :param HW: height & width, h = w
        :param kernel: kernel size
        '''
        return (((HW-kernel + 2*0)/stride)+1)

    def forward(self, x):
        #Convolution-Pooling block 1:
        x = self.conv1(x)
        x = self.mpool1(x)
        #Convolution-Pooling block 2:
        x = self.conv2(x)
        x = self.mpool2(x)
        #Convolution-Pooling block 3:
        x = self.conv3(x)
        x = self.mpool3(x)

        #Outputs
        x = self.flatten(x)
        x = self.fco(x)

