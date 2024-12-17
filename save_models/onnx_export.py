import torch
from modelC3 import CNN
import PIL
import torchvision.transforms.v2 as v2

#model
model_path = 'models/2024-09-30 18_13_15.758843'
state_dict = torch.load(model_path,map_location=torch.device('cpu'))
model = CNN(1,32,5,2,64,5,2,128,5,2,256)
model.load_state_dict(state_dict=state_dict)

#input
img_path = 'IMG_8171.JPG'
image = PIL.Image.open(img_path)
image = image.convert('L')

resize_transform = v2.Resize([28,28])
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
#export
onnx_path = 'onnx_' + model_path+'.onnx'
torch.onnx.export(model,transformed_img,onnx_path)