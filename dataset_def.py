from numpy import ndarray
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms.v2 as v2
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST

def load_csv_data(csv_path:str):
    data_df = pd.read_csv(csv_path)
    labels = (data_df.iloc[:,0]).values #first column, labels are uppercase & lowecase, e.g "a" or "A" = 1, shift range from 1-26 to 0-25
    raw_pixels = data_df.iloc[:,1:].values #pixel values , transpose to get correct orientation

    imgs = raw_pixels.reshape(-1,28,28) #emnist images are 28*28 pixels - 3d ndarray
    return imgs, labels

class EMNISTDataset(Dataset):
    def __init__(self, img_data:ndarray, label_data:ndarray, transform, target_transform, subset_indices:slice,):
        super().__init__()
        self.img_subset = img_data[subset_indices]
        self.label_subset = label_data[subset_indices]

        if self.img_subset.shape[0] != self.label_subset.shape[0]:
            raise RuntimeError(f"Img subset len does not match label subset len: \n Img subset len: {self.img_subset.shape[0]} \n Label subset len:{self.label_subset.shape[0]}")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        data_len = self.label_subset.shape[0]
        return data_len
    
    def __getitem__(self, idx):
        X = self.img_subset[idx,:,:]
        y = int(self.label_subset[idx])
        if self.transform:
            X = self.transform(X) #applies torchvision transform
        if self.target_transform:
            y = self.target_transform(y) #applies torchvision transform
        return X, y
        
def show_images(img:torch.Tensor, label):
        """ Function to display images and their labels """
        img = img.permute(1,2,0)
        print(img.shape)
        img_np = img.numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np,cmap='gray')
        plt.axis('off')
        plt.title(label=label)
        plt.show()
        
if __name__ == '__main__':
    img_transform = v2.Compose([
    v2.Resize(28),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
    v2.RandomHorizontalFlip(p=1),#100% probability
    v2.RandomRotation(degrees=(90,90)), #flip 90 degrees
])
    imgs, labels = load_csv_data('EMNIST_data/emnist-letters-train.csv')
    subset_slice = slice(0,5000)
    own_dataset = EMNISTDataset(imgs,labels,transform=img_transform,target_transform=lambda y:y-1,subset_indices=subset_slice)
    own_img, own_label = own_dataset.__getitem__(0)

    torch_dataset = EMNIST('torch_EMNIST',split='letters',train=True,download=True, transform=img_transform, target_transform = lambda y:y-1,)
    torch_img, torch_label = torch_dataset.__getitem__(0)
    show_images(own_img,own_label)
    show_images(torch_img,torch_label)
    print('Dataset tested')


