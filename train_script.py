from torchvision.transforms import v2
import json
from torch.utils.data import random_split
from model_def import CNN
from dataset_def import load_csv_data, EMNISTDataset
import torch
import wandb

n_runs = 10

def read_config(config_path):
    with open(config_path) as conf_file:
        config_dict = json.load(conf_file)
        return config_dict 

config_dict = read_config('config.json')

#augment image
image_transform = v2.Compose( 

)

for i in range(n_runs):
    #set random seeds for reproducibility
    seed = 100*i + i
    torch.seed(seed)
    torch.cuda.seed_all(seed)
    torch.mps.seed(seed)

    #init wandb run
    wandb.init(name = djqd0q, config=config_dict,job_type='experimental run')
    #import train data
    train_subset_size = config_dict['train_size']
    test_subset_size = config_dict['test_size']

    


    model = CNN(
        ...
    )