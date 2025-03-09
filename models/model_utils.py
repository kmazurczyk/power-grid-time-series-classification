from copy import deepcopy
from itertools import product
import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from __init__ import get_base_path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

random_seed = int(os.getenv('RANDOM_SEED'))
torch.manual_seed(random_seed)

class FeatureStore:
    def __init__(self, data):
        self.data = data
        self.R1_RMS = [i for i in data.columns if 'r1' in i.lower() and 'magnitude' in i.lower() and 'zero' not in i.lower()]
        self.R1_waves = [i for i in data.columns if 'r1' in i.lower() and 'wave' in i.lower() and 'zero' not in i.lower()]
        self.R1_R2_RMS = [i for i in data.columns if ('r1' in i.lower() or 'r2' in i.lower()) and 'magnitude' in i.lower() and 'zero' not in i.lower()]
        self.R1_R2_waves = [i for i in data.columns if ('r1' in i.lower() or 'r2' in i.lower()) and 'wave' in i.lower() and 'zero' not in i.lower()]
        self.all_power_waves = [i for i in data.columns if 'wave' in i.lower() and 'power' in i.lower() and 'zero' not in i.lower()]
        self.y_binary = 'is_attack'
        self.y_tertiary = 'scenario_class'
        self.y_broad_class = 'scenario_broad_type'
        self.y_full_class = 'scenario_type'

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


class SK_Classification_Experiment:
    '''
    loop through feature and model selection for SKLearn Classifiers
    returns list of length (x_features * y_features * n_estimators)
    list items are dict() of train/test classification scores and a deepcopy of the estimator for access to attributes
    '''

    def __init__(self, data:pd.DataFrame, X_feature_list:list, y_feature_list:list, estimators:list, splitter=timeseries_train_test_split, scaler=StandardScaler, target_encoder=LabelEncoder(), random_seed=random_seed):
        self.data = data
        self.X_feature_list = X_feature_list
        self.y_feature_list = y_feature_list
        self.estimators = estimators
        self.splitter = splitter
        self.scaler = scaler()
        self.target_encoder = target_encoder
        self.random_seed = random_seed
        self.experiment_scores = []
    
    def permute_features(self):
        return product(self.X_feature_list, self.y_feature_list)

    def train_test_split(self,x_features,y_feature):
        X = self.data.loc[:,x_features]
        y = self.data.loc[:,y_feature]
        return self.splitter(X, y)

    def run_experiments(self):
        # permute features
        for x, y in self.permute_features():
            # train test split
            X_train, y_train, X_test, y_test = self.train_test_split(x,y)
            
            # target encoding
            if self.target_encoder is not None:
                enc = self.target_encoder.fit(y_train)
                y_train, y_test = enc.transform(y_train), enc.transform(y_test)
                labels = [i for i,j in enumerate(enc.classes_)]
                target_names = [str(i) for i in enc.classes_]
            else:
                labels = [i for i,j in enumerate(self.data[y].unique())]
                target_names = self.data[y].unique()

            # preprocessor
            if isinstance(X_train,pd.DataFrame):
                numeric_features = X_train.select_dtypes('number').columns
            elif isinstance(X_train,pd.Series) and is_numeric_dtype(X_train):
                numeric_features = [X_train.name]
            else: 
                print('No numeric features found in X')

            preprocessor = ColumnTransformer(
                transformers=[('scaler', self.scaler, numeric_features)], remainder='passthrough')
        
            for estimator in self.estimators:
                # train
                pipe = Pipeline([
                        ('preprocessor', preprocessor),
                        ('estimator', estimator)
                    ])

                pipe.fit(X_train, y_train)

                # test
                train_pred, test_pred = pipe.predict(X_train), pipe.predict(X_test)

                # classification scores
                self.experiment_scores += [{
                    'X_features':x,
                    'y_features':y,
                    'y_classes':target_names,
                    'estimator':deepcopy(estimator),
                    'train_true':y_train,
                    'train_pred':train_pred,
                    'test_true':y_test,
                    'test_pred':test_pred,
                    'training_classification_report':classification_report(y_train, train_pred, labels=labels, target_names=target_names),
                    'training_confusion_matrix':multilabel_confusion_matrix(y_train, train_pred, labels=labels),
                    'train_recall':recall_score(y_train, train_pred, labels=labels, average='macro'),
                    'train_precision':precision_score(y_train, train_pred, labels=labels, average='macro'),
                    'train_f1':f1_score(y_train, train_pred, labels=labels, average='macro'),
                    'train_accuracy':accuracy_score(y_train, train_pred),
                    'test_classification_report':classification_report(y_test, test_pred, labels=labels, target_names=target_names),
                    'test_confusion_matrix':multilabel_confusion_matrix(y_test, test_pred, labels=labels),
                    'test_recall':recall_score(y_test, test_pred, labels=labels, average='macro'),
                    'test_precision':precision_score(y_test, test_pred, labels=labels, average='macro'),
                    'test_f1':f1_score(y_test, test_pred, labels=labels, average='macro'),
                    'test_accuracy':accuracy_score(y_test, test_pred)
                        }]

def pad_collate(batch):
    """padding function that will be used for dataloader parameter collate_fn
    ref https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html"""

    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True)
    yy_pad = pad_sequence(yy, batch_first=True)

    return xx_pad, yy_pad, x_lens, y_lens

class SignalClassificationDataset(Dataset):
    def __init__(self, signals, labels, device, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx],dtype=torch.float).to(self.device)
        label = torch.tensor(self.labels[idx],dtype=torch.long).to(self.device)

        # transform
        if self.transform is not None:
            signal, label = self.transform(signal,label)

        # signal to tensor
        return signal, label
    
class AugmentMinorityClass(object):
    """augment minority classes with strided samples that have some noise added"""
    def __init__(self, target_classes, device, stride=None):
        self.target_classes = target_classes
        self.stride = stride
        self.device = device

    def __call__(self, signal, label):
        signal_len, feature_len = signal.size()
        
        if self.stride == None:
            self.stride = (1,feature_len)               # move down one row at a time, across features
        window_size = (len(signal) // 3, feature_len)   # about 1/3 size of the data
        
        if label in self.target_classes:
            # stride
            signal = torch.as_strided(signal,
                                      size= window_size,
                                      stride = self.stride).to(self.device)

            # add some noise
            signal = signal + (torch.randn(signal.size(0),signal.size(1),device=self.device)*0.01)

        return signal, label          

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=1, num_layers=1, drop_out=0, output_size=1):
        super().__init__()

        # data attributes
        self.input_size = input_size # number of features
        self.output_size = output_size # number of classes for y

        # model attributes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = drop_out
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers, 
                            batch_first = True, 
                            dropout=drop_out)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

        # model evaluation
        self.train_logits = []
        self.train_predictions = []
        self.train_loss_epoch = []
        self.train_accuracy_epoch = []

        self.test_logits = []
        self.test_loss = None
        self.test_accuracy = None

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

    def lstm_train(self, dataloader, criterion, optimizer, n_epochs=5):
        # for evaluation
        n_samples = 0
        n_correct = 0

        for epoch in range(n_epochs):
            for i, (x, y, x_lens, y_lens) in enumerate(dataloader):
                # reset gradients
                optimizer.zero_grad()
                y = y.view(-1)
                
                # forward
                out, __ = self(x, x_lens)
                loss = criterion(out, y)
            
                # back
                loss.backward()
                optimizer.step()
           
                # evaluation
                __, predicted = torch.max(out, 1)
                correct = (predicted == y).sum().item()
                n_samples += y.size(0)
                n_correct += correct      

                self.train_logits += [out]
                self.train_predictions += [predicted]

                if i % 10 == 0:
                    print(f"Epoch: {epoch+1}/{n_epochs}, Step {i+1}, Training Loss {loss.item():.4f}, Accuracy {n_correct/n_samples:.4f}")
            
            # track performance per epoch
            self.train_loss_epoch += [loss]
            self.train_accuracy_epoch += [n_correct / n_samples]

    def lstm_test(self, dataloader, criterion):
        pass
            
if __name__ == "__main__":
    pass 
