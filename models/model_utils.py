from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from __init__ import get_base_path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, RocCurveDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

random_seed = int(os.getenv('RANDOM_SEED'))
torch.manual_seed(random_seed)
rnn_data_dir = get_base_path() + os.getenv('RNN_DATA_DIR')

class FeatureStore:
    def __init__(self, data):
        self.data = data
        self.R1_waves = [i for i in data.columns if 'r1' in i.lower() and 'wave' in i.lower()]
        self.R1_R2_waves = [i for i in data.columns if ('r1' in i.lower() or 'r2' in i.lower()) and 'wave' in i.lower()]
        self.y_binary = 'is_attack'
        self.y_tertiary = 'scenario_class'
        self.y_broad_class = 'scenario_type'
        self.y_full_class = 'marker'

    def __str__(self):
        print(f'''Feature Store giving column names for a dataframe 
        with X features: 
        R1_waves {self.R1_waves}
        R1_R2_waves {self.R1_R2_waves} 
        and predictors:
        y_binary {self.y_binary}
        y_tertiary {self.y_tertiary}
        y_broad_class {self.y_broad_class}
        y_full_class {self.y_full_class}''')

# ClassificationModel class with scoring
class ClassificationModel:
    def __init__(self, model, config: dict):
        self.model = model
        self.metrics = []
        self.config = config
    
    def __str__(self):
        return f"Model: {self.model} configured with {self.config} metrics {self.metrics}"

    def train(self, X, y) -> None:
        return self.model.fit(X)

    def predict(self, X) -> np.array:
        return self.model.predict(X)

    def accuracy_score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def confusion_matrix_from_model(self, X, y, y_labels:list, display=True):
        if display:
            ConfusionMatrixDisplay.from_estimator(
                self.model,
                X,
                y,
                display_labels=y_labels,
                normalize='all',
            )
            plt.title(f'Confusion matrix {self.model}, {y.columns()} (normalized)')
            plt.show()
        return confusion_matrix(self.predict(X), y, labels=y_labels)

    def roc_curve(self, X, y, display=True):
        pass

    def classification_report_from_model(self, X, y, display=True):
        c = classification_report(self.predict(X), y, labels=y_labels)
        if display:
            print(c)
        return c

def timeseries_train_test_split(X,y,test_size=None):
    '''
    Requires:
    X features :: pd.DataFrame
    y features :: pd.Series
    test_size = proportion of data :: float

    Returns:
    Returns X_train, y_train, X_test, y_test dataframes using TimeSeriesSplit
    '''
    if test_size == None:
        test_size = len(X)//5
    else:
        test_size = round(test_size * len(sample_bag))

    timetraintestsplit = TimeSeriesSplit(n_splits=2,test_size=test_size)
    splits = timetraintestsplit.split(X,y)
    train_indices, test_indices = list(splits).pop()
    X_train, y_train, X_test, y_test =  X.iloc[train_indices], y.iloc[train_indices], \
                                        X.iloc[test_indices], y.iloc[test_indices]
    return X_train, y_train, X_test, y_test

def grab_bag_train_test_split(X,y,id_series='sample_id',test_size=None,return_ids=False):
    '''
    Requires:
    X features :: pd.DataFrame
    y features :: pd.Series
    id_series :: pd.Series
    test_size = proportion of data :: float

    Randomly samples unique ids for train and test
    
    Returns:
    X_train, y_train, X_test, y_test dataframes and optionally sample_id arrays'''

    np.random.seed(seed=random_seed)

    # SAMPLE
    sample_bag = id_series.unique()
    
    if test_size == None:
        test_size = len(sample_bag)//5
    else:
        test_size = round(test_size * len(sample_bag))
    
    test_ids = np.random.choice(sample_bag, size=test_size, replace=False)
    train_ids = np.setdiff1d(sample_bag,test_ids)

    X_train, y_train, X_test, y_test =  X.loc[id_series.isin(train_ids)], y.loc[id_series.isin(train_ids)], \
                                        X.loc[id_series.isin(test_ids)], y.loc[id_series.isin(test_ids)]


    X_train, y_train, X_test, y_test =  X.loc[id_series.isin(train_ids)], y.loc[id_series.isin(train_ids)], \
                                        X.loc[id_series.isin(test_ids)], y.loc[id_series.isin(test_ids)]
    
    if return_ids:
        return X_train, y_train, X_test, y_test, train_ids, test_ids
    else:
        return X_train, y_train, X_test, y_test

# padding function that will be used for dataloader parameter collate_fn
# syntax from https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html

def pad_collate(batch):
    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True)
    yy_pad = pad_sequence(yy, batch_first=True)

    return xx_pad, yy_pad, x_lens, y_lens

# class NonZeroLabelEncoder:
#     '''because we are going to use zero padding, we can't have any class encodings == 0'''
#     def __init__(self):
#         self.classes = None
#         self.encodings = None
#         self.mappings = None

#     def fit(self,y) -> None:
#         self.classes = np.sort(np.unique(y))
#         self.mappings = tuple(enumerate(self.classes,start=1))
#         self.encodings = [i[0] for i in self.mappings]

#     def transform(self,y) -> np.array:
#         mappings = {i[1]:i[0] for i in self.mappings}
#         y = y.map(mappings)
#         return y.to_numpy()
    
#     def reverse_transform(self,y) -> pd.Series:
#         mappings = {i[0]:i[1] for i in self.mappings}
#         y = pd.Series(y)
#         y = y.map(mappings)
#         return y

class SignalClassificationDataset(Dataset):
    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # transform
        if self.transform is not None:
            signal = self.transform(signal)

        # signal to tensor
        return torch.tensor(signal,dtype=torch.float), torch.tensor(label,dtype=torch.long)

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=1, num_layers=1, dropout=0, output_size=1):
        super().__init__()

        # data attributes
        self.input_size = input_size # number of features
        self.output_size = output_size # number of classes for y

        # model attributes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers, 
                            batch_first = True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, x_lens):
        # PACK
        packed_x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # LSTM
        #   x = pack_padded_sequence(x, x_lens)
        #   optional h0 = torch.zeroes(self.num_layers, packed_x.size(0), self.hidden_size)
        #   out = (batch, input_size, hidden_size)
        packed_out, (hn, cn) = self.lstm(packed_x)

        # UNPACK
        out_pad, out_lens = pad_packed_sequence(packed_out, batch_first=True)

        # FULLY CONNECTED LAYER 
        out = out_pad[:, -1, :] # get last state from LSTM
        out = self.fc(out)

        # CLASS PREDICTION - raw logits should go to CrossEntropyLoss()
        return out, (hn, cn)

if __name__ == "__main__":
    pass 
