from torchvision.transforms import v2
import json
from torch.utils.data import random_split, DataLoader
from model_def import CNN
from dataset_def import load_csv_data, EMNISTDataset
import torch
import wandb
from trainer_def import CNNTrainer

n_runs = 10

def read_config(config_path):
    with open(config_path) as conf_file:
        config_dict = json.load(conf_file)
        return config_dict 

config_dict = read_config('config.json')

#augment image
image_transform = v2.Compose([
    v2.Resize(28),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
    v2.Normalize(mean=(0.1736,), std=(0.3248,))
]

)
train_imgs, train_labels = load_csv_data('EMNIST_l/emnist-letters-train.csv')
test_imgs, test_labels = load_csv_data('EMNIST_l/emnist-letters-test.csv')

for i in range(n_runs):
    #set random seeds for reproducibility
    seed = 100*i + i
    torch.seed(seed)
    torch.cuda.seed_all(seed)
    torch.mps.seed(seed)

    train_subset_size = config_dict['train_size']
    test_subset_size = config_dict['test_size']
    #init wandb run
    run = wandb.init(name = f"Run-D{train_subset_size}-S{seed}", config=config_dict,job_type='experimental run')
    
    #import train data
    train_indices = slice(0,train_subset_size)
    train_dataset = EMNISTDataset(
        train_imgs,
        train_labels,
        transform=image_transform,
        subset_indices=train_indices)
    
    #import test data:
    test_indices = slice(0,test_subset_size)
    test_dataset = EMNISTDataset(
        test_imgs,
        test_labels,
        transform=image_transform,
        subset_indices=test_indices
    )

    batch_size = run.config['batch_size']

    n_workers = 0
    #dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True, num_workers=n_workers)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True, num_workers=n_workers)


    conv1= run.config['conv1']
    ckernel1= run.config['ckernel1']
    MPkernel1= run.config['MPkernel1']
    conv2= run.config['conv2']
    ckernel2= run.config['ckernel2']
    MPkernel2= run.config['MPkernel2']
    conv3= run.config['conv3']
    ckernel3= run.config['ckernel3']
    MPkernel3= run.config['MPkernel3']

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
        out_dim=26,
        input_HW=28
    )

    trainer = CNNTrainer(run=run,
                         model = model,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,)
    CNNTrainer.full_epoch_loop()
    run.finish(0)

