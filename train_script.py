
import torchvision
from torchvision.transforms import v2
import json
from torch.utils.data import random_split

def read_config(config_path):
    with open(config_path) as conf_file:
        config_dict = json.load(conf_file)
        return config_dict 

config_dict = read_config('config.json')

#augment image
image_transform = v2.Compose( 

)
#import train data
full_train_dataset = torchvision.datasets.EMNIST(
    root='EMNIST_data/train',
    download = True,
    train = True,
    split='letters',
    transform = image_transform
)

full_test_dataset = torchvision.datasets.EMNIST(
    root='EMNIST_data/train',
    download = True,
    train = False,
    split='letters',
    transform = image_transform
)

subset_p = config_dict['subset_p']


