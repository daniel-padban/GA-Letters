import PIL
import PIL.ImageOps
from matplotlib import axes
import torch
from model_def import CNN
import torch.nn as nn
import json
import PIL
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np


device = ( #selects device
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
device='cpu'
print(device)
n_config_dicts = 1

def read_config(config_path): #load config dictionary
    with open(config_path) as conf_file:
        config_dict = json.load(conf_file)
        return config_dict 

config_dict = read_config('config.json')

conv1 = config_dict['conv1']
ckernel1 = config_dict['ckernel1']
MPkernel1 = config_dict['MPkernel1']
conv2 = config_dict['conv2']
ckernel2 = config_dict['ckernel2']
MPkernel2 = config_dict['MPkernel2']
conv3 = config_dict['conv3']
ckernel3 = config_dict['ckernel3']
MPkernel3 = config_dict['MPkernel3']
fc1 = config_dict['fc1']

model = CNN(
    input_channels=1,
    conv1=conv1,
    ckernel1=ckernel1,
    MPkernel1=MPkernel1,
    conv2=conv2,
    ckernel2=ckernel2,
    MPkernel2=MPkernel2,
    conv3=conv3,
    ckernel3=ckernel3,
    MPkernel3=MPkernel3,
    fc1=fc1,
    out_dim=26,
    input_HW=28,
    device=device
)

model_timestamp = '2024-09-30 11_11_20.042451'
path = f'models/{model_timestamp}'
state_dict = torch.load(path,map_location=torch.device(device),weights_only=True)
model.load_state_dict(state_dict)
model.eval()

resize_transform = v2.Resize(28)

img_path = 'imgs/W_emnist.jpeg'
image = PIL.Image.open(img_path)
image = image.convert('L')
image = resize_transform(image)
inverted_img = PIL.ImageOps.invert(image)
inverted_img = resize_transform(inverted_img)

image_transform = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
    v2.RandomHorizontalFlip(p=1),#100% probability
    v2.RandomRotation(degrees=(90,90)), #flip 90 degrees
    v2.Normalize((0.1736,),(0.3248,))
])
transformed_img:torch.Tensor = image_transform(inverted_img)
transformed_img = transformed_img.unsqueeze(0)

logits = model(transformed_img)
probabilities = torch.softmax(logits,1)
most_probable = torch.argmax(probabilities).item()

fig,(ax1,ax2) = plt.subplots(1,2)
fig.col
ax1:axes.Axes
ax2:axes.Axes

ax1.imshow(image, cmap='gray')
#ax1.axis(xmin=0,xmax=28,ymin=0,ymax=28)
ax1.set_label('Original Image')

ax2.imshow(inverted_img,cmap='gray')
#ax2.axis(xmin=0,xmax=28,ymin=0,ymax=28)
ax2.set_label('Inverted image')
plt.show()
print(most_probable)