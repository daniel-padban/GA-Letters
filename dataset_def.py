from numpy import ndarray
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def load_csv_data(csv_path:str):
    data_df = pd.read_csv(csv_path)
    labels = (data_df.iloc[:,0]-1).values #first column, labels are uppercase & lowecase, e.g "a" or "A" = 1, shift range from 1-26 to 0-25
    raw_pixels = data_df.iloc[:,1:].values #pixel values , transpose to get correct orientation

    imgs = raw_pixels.reshape(-1,28,28) #emnist images are 28*28 pixels - 3d ndarray
    return imgs, labels

class EMNISTDataset(Dataset):
    def __init__(self, img_data:ndarray, label_data:ndarray, transform, subset_indices:slice,):
        super().__init__()

        self.img_subset = img_data[subset_indices]
        self.label_subset = label_data[subset_indices]

        if self.img_subset.shape[0] != self.label_subset.shape[0]:
            raise RuntimeError(f"Img subset len does not match label subset len: \n Img subset len: {self.img_subset.shape[0]} \n Label subset len:{self.label_subset.shape[0]}")

        self.transform = transform

    def __len__(self):
        data_len = self.label_subset.shape[0]
        return data_len
    
    def __getitem__(self, idx):
        X = self.img_subset[idx,:,:]
        y = self.label_subset[idx]
        if self.transform:
            X = self.transform(X) #applies torchvision transform
        return X, y
        
        


