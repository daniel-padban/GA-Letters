import PIL.Image
import PIL.ImageOps
import torch
from model_def import CNN
import json
import PIL
import torchvision.transforms.v2 as v2

device = ( #selects device
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
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

model_timestamp = ''
path = f'models/{model_timestamp}'
state_dict = torch.load(path)
model = CNN.load_state_dict(state_dict=state_dict)
model.eval()

img_path = 'imgs/wfwwefwfweif'
image = PIL.Image.open(img_path)
inverted_img = PIL.ImageOps.invert(image)
image_transform = v2.Compose([
    v2.Resize(28),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
    v2.RandomHorizontalFlip(p=1),#100% probability
    v2.RandomRotation(degrees=(90,90)), #flip 90 degrees
    v2.Normalize((0.1736,),(0.3248,))
])
transformed_img = image_transform(inverted_img)

logits = model(transformed_img)
probabilities = torch.softmax(logits)
most_probable = torch.argmax(probabilities)


print(most_probable)