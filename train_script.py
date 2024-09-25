from cProfile import label
import torch.utils
import torch.utils.data
from torchvision.transforms import v2
import json
from torch.utils.data import random_split, DataLoader
from model_def import CNN
from dataset_def import load_csv_data, EMNISTDataset
import torch
import wandb
from trainer_def import CNNTrainer
import matplotlib.pyplot as plt
from init_weights import init_model_w

device = ( #selects device
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
print(device)

n_runs = 10
def show_images(dataset:torch.utils.data.Dataset):
        """ Function to display images and their labels """
        img_tensor, label= dataset.__getitem__(0)
        img_tensor = img_tensor.permute(1,2,0)
        print(img_tensor.shape)
        label = label
        img_np = img_tensor.numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np,cmap='gray')
        plt.axis('off')
        plt.title(label=label)
        plt.show()

def read_config(config_path): #load config dictionary
    with open(config_path) as conf_file:
        config_dict = json.load(conf_file)
        return config_dict 

config_dict = read_config('config.json')

#define transform to augment image
image_transform = v2.Compose([
    v2.Resize(28),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
    v2.RandomHorizontalFlip(p=1),#100% probability
    v2.RandomRotation(degrees=(90,90)), #flip 90 degrees
    v2.Normalize((0.1736,),(0.3248,))
])
train_imgs, train_labels = load_csv_data('EMNIST_data/emnist-letters-train.csv')
test_imgs, test_labels = load_csv_data('EMNIST_data/emnist-letters-test.csv')
#labels are shifted -1 step, range changes from 1-26 to 0-25, (needed for loss calculation)

for i in range(n_runs):
    #set random seeds for reproducibility
    seed = 100*i + i
    print(f'---------- Run {seed} ----------')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)

    train_subset_size = config_dict['train_size']
    test_subset_size = config_dict['test_size']
    #init wandb run
    run = wandb.init(name = f"Run-D{train_subset_size}-S{seed}-Resnet18", config=config_dict,job_type='experimental run')
    
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

    #get hyperparams from config
    conv1= run.config['conv1']
    ckernel1= run.config['ckernel1']
    MPkernel1= run.config['MPkernel1']
    conv2= run.config['conv2']
    ckernel2= run.config['ckernel2']
    MPkernel2= run.config['MPkernel2']

    model = CNN(
        input_channels=1,
        conv1=conv1,
        ckernel1=ckernel1,
        MPkernel1=MPkernel1,
        conv2=conv2,
        ckernel2=ckernel2,
        MPkernel2=MPkernel2,
        out_dim=26,
        input_HW=28
    )
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) #resnet18

    model.to(device=device)
    #init_model_w(model=model)

    #train
    trainer = CNNTrainer(run=run,
                         model = model,
                         device=device,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,)
    trainer.full_epoch_loop(print_gradients=True)
    run.finish(0)

