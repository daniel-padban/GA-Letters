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
                #block 1
                conv1:int,
                ckernel1:int,
                MPkernel1:int,
                #block 2
                conv2:int,
                ckernel2:int,
                MPkernel2:int,
                #block 3
                conv3:int,
                ckernel3:int,
                MPkernel3:int,
                
                fc1:int,

                out_dim:int=26,
                input_HW:int = 28,
                device = 'cpu',
                ):
        super().__init__()
        
        #Convolution-Pooling block 1:
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=conv1,kernel_size=ckernel1,padding='same', device=device)
        #conv2_HW = self.calculate_dims(input_HW,ckernel2,stride=1) #HW after conv1 layer
        #self.bn1 = nn.BatchNorm2d(conv1,)
        self.activation1 = nn.SiLU()
        self.mpool1 = nn.MaxPool2d(kernel_size=MPkernel1)
        mpool1_HW = self.calculate_dims(input_HW,MPkernel1,stride=MPkernel1) #height and/or width of mpool1 output

        #Convolution-Pooling block 2:
        self.conv2 = nn.Conv2d(in_channels=conv1,out_channels=conv2,kernel_size=ckernel2,padding='same',device=device)
        #self.bn2 = nn.BatchNorm2d(conv2,)
        self.activation2 = nn.SiLU()
        self.mpool2 = nn.MaxPool2d(kernel_size=MPkernel2)
        mpool2_HW = self.calculate_dims(mpool1_HW,MPkernel2,stride=MPkernel2) #height and/or width of mpool2 output

        #Convolution-Pooling block 3:
        self.conv3 = nn.Conv2d(in_channels=conv2,out_channels=conv3,kernel_size=ckernel3,padding='same',device=device)
        #self.bn3 = nn.BatchNorm2d(conv3,)
        self.drop2d1 = nn.Dropout2d()
        self.activation3 = nn.SiLU()
        self.mpool3 = nn.MaxPool2d(kernel_size=MPkernel3,stride=1) #input 7 –> output 6
        mpool3_HW = self.calculate_dims(mpool2_HW,MPkernel3,stride=1) #height and/or width of mpool3 output

        #output block
        self.flatten = nn.Flatten(1,-1)
        flattened_dim = int((mpool3_HW**2)*conv3)
        self.activation3 = nn.SiLU()
        self.fc1 = nn.Linear(in_features=flattened_dim,out_features=fc1, device=device)
        self.drop1d1 = nn.Dropout1d() #output dropout
        self.fc2 = nn.Linear(in_features=fc1,out_features=out_dim, device=device)

    def calculate_dims(self, HW:int, kernel:int,stride:int=1,):
        '''
        :param HW: height x width, h = w
        :param kernel: kernel size
        '''
        return (((HW-kernel + 2*0)/stride)+1)

    def forward(self, x:torch.Tensor):
        x.requires_grad_()

        #Convolution-Pooling block 1:
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.mpool1(x)
        
        #Convolution-Pooling block 2:
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.mpool2(x)

        x = self.conv3(x)
        x = self.drop2d1(x)
        x = self.activation3(x)
        x = self.mpool3(x)
        
        #Outputs
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.drop1d1(x)
        output = self.fc2(x)
        return output

