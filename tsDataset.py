from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset
from TSds import TSds

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class TimeseriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])
    
    
    
class TimeseriesDatasetDeepAnt(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len].permute(1,0), self.y[index+self.seq_len-1])
    
    

class TimeSeriesDatasetAE(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len].permute(1,0), self.y[index+self.seq_len-1])


def get_from_one(ts, window_size, stride):
    ts_length = ts.shape[0]
    samples = []
    for start in np.arange(0, ts_length, stride):
        if start + window_size > ts_length:
            break
        samples.append(ts[start:start+window_size])
    return np.array(samples)

def load_data(train_x, test_x, test_y, val_size: int, window_size:int = 100, stride:int = 1, batch_size: int= 64, dataloader:bool = False):

    nc = train_x.shape[1]
    train_len = int(len(train_x) * (1-val_size))
    val_x = train_x[train_len:]
    train_x = train_x[:train_len]


    print('Training data:', train_x.shape)
    print('Validation data:', val_x.shape)
    print('Testing data:', test_x.shape)
    
    if dataloader:
        train_x  = get_from_one(train_x, window_size, stride)
        train_y = np.zeros(len(train_x))

        train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))

        data_loader = {
            "train": DataLoader(
                dataset = train_dataset,
                batch_size = batch_size,
                shuffle = False,
                num_workers= 0,
                drop_last=False
                ),
            "val" : val_x,
            "test":(test_x, test_y),
            "nc": nc
            }

        return data_loader



