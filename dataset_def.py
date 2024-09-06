import torch
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self,):
        super().__init__()
        #init and define variables

    def get_data(self): #load data from source
        
        return data
    
    def process_data(self): #normalize etc.
        
        return processed_X, processed_y

    def __len__(self): # get length of dataset
        return data_len
    
    def __getitem__(self, idx): #get each image and label

        return X,y
     
    



