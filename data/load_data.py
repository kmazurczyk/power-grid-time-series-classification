
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import re
from datetime import timedelta, datetime
from __init__ import get_base_path
from scipy.io import arff

load_dotenv()
raw_data_path = get_base_path()+ os.getenv('RAW_DATA_DIR')
clean_data_path = get_base_path() + os.getenv('CLEAN_DATA_DIR')
label_file = raw_data_path + os.getenv('LABEL_FILE')
marker_file = raw_data_path + os.getenv('MARKER_FILE')
combined_csv = clean_data_path + os.getenv('COMBINED_DATA_FILE')

class DataLoader:

    def __init__(self):
        pass

    def __str__(self):
        print("DataLoader", dir(DataLoader))

    def load_arff_data(self, data_path: str) -> pd.DataFrame:

        # LOCATE EACH ARFF FILE
        data_files = [f for root, dirs, files in os.walk(data_path) for f in files if 'data' in f]

        if len(data_files) == 0:
            raise Exception(f"No data files found at the following path: {data_path}")

        # SORT FILES BY FILE NAME
        data_files.sort(key=lambda x: int(re.findall('[0-9]{1,}',x)[0])) # regex gets int from each file name
      
        # ARFF TO PANDAS
        dfs = [arff.loadarff(data_path + f)[0] for f in data_files]
        dfs = [pd.DataFrame(i) for i in dfs]

        # ADD SOURCE FILE INDEX
        for i in range(0,len(dfs)):
            dfs[i]['source_file'] = i+1

        # COMBINE DFS
        dfs = pd.concat(dfs,ignore_index=True)   
        return dfs
    
    def apply_column_names(self, label_file: str, df: pd.DataFrame) -> pd.DataFrame:
        labels = pd.read_csv(label_file)
        labels_dict = dict(zip(labels['Source'],labels['Target']))
        df = df.rename(columns = labels_dict)
        return df

class DataPreprocessor:

    def __init__(self, data):
        self.data = data
        self.columns = data.columns.to_list()
        self.random_seed = int(os.getenv('RANDOM_SEED'))
        # self.scaler = StandardScaler()

    def __str__(self):
        pass

    # TO DO imputation columns for inferance
    #       refactor preprocessing to map columns to dtypes
    #       scaling
    #       encoding

    def apply_event_labels(self, marker_file: str, key: str):
        markers = pd.read_csv(marker_file)
        self.data = self.data.merge(right=markers, how='left', on=key)
        return self
    
    def binarize_class_feature(self, feature_name: str, new_name: str, criteria: list):
        # labels 1 for class labels found in list else 0
        criteria = [str.lower(i) for i in criteria]
        self.data[new_name] = self.data[feature_name].apply(lambda x: 1 if x.lower() in criteria else 0)
        return self

    def generate_sample_ids(self, unique_columns=['marker','source_file']):
        # if the class marker or file number changes, consider it the start of a new sample and increment id
        np.random.seed(self.random_seed)
        key = np.random.randint(self.data.shape[0],high=None)
        sample_ids = []
        for i, s in self.data.iterrows():
            if i != 0:
                criteria = (self.data.loc[i,unique_columns] == self.data.loc[i-1,unique_columns])
                if criteria.all() == False:
                    key += 1
            sample_ids += [key]
        self.data['sample_id'] = sample_ids
        return self
    
    def generate_time_series(self):
        # data are not timestamped but are sequential, so we'll synthesize a timeseries
        synthetic_datetime = []

        # instantiate a start date & time
        np.random.seed(self.random_seed)
        hour, min, sec = [np.random.randint(low=0, high=i) for i in (24,60,60)]
        timestamp = datetime(1970, 1, 1, hour, min, sec)

        # determine the offsets
        max_sample_length = self.data.loc[:,['sample_id','marker']].groupby('sample_id').count().max().to_numpy()[0]
        min_increment = int(max_sample_length // 120 + 1)
        delta = timedelta(seconds=1/120)            # time increment within the sample
        offset = timedelta(seconds=min_increment)   # offset between samples

        # partition the samples
        sample_id_counts = self.data['sample_id'].value_counts().to_dict()

        # increment time
        for s, n in sample_id_counts.items():
            timestamp += offset                     # offset isn't working
            for i in range(n):
                timestamp += delta
                synthetic_datetime += [timestamp]

        self.data['synthetic_datetime'] = synthetic_datetime
        self.data['synthetic_datetime'] = pd.to_datetime(self.data['synthetic_datetime'])
        return self

    def preprocess(self):
        # fix numeric typing
        relays = ['R1','R2','R3','R4']
        for r in relays:
            self.data[f'{r}_appearance_impedance_for_relays'] = self.data[f'{r}_appearance_impedance_for_relays'].astype('float64')

        # fix byte string on marker class
        self.data['marker'] = self.data['marker'].astype('int').astype('category')

        # fix datetime
        self.data['synthetic_datetime'] = pd.to_datetime(self.data['synthetic_datetime'])
        return self

    def get_dataframe(self) -> pd.DataFrame:
        return self.data


if __name__ == '__main__':
    if os.path.exists(combined_csv):
        print(f"The expected data file already exists at path: {combined_csv}")
    else:
        # LOAD DATA
        dl = DataLoader()
        df = dl.load_arff_data(raw_data_path)
        df = dl.apply_column_names(label_file, df)

        # CLEAN DATA
        dp = DataPreprocessor(df)
        df = dp.preprocess() \
            .apply_event_labels(marker_file, 'marker') \
            .binarize_class_feature('scenario_class', 'is_attack', ['attack']) \
            .generate_sample_ids() \
            .generate_time_series() \
            .get_dataframe()
        df.to_csv(combined_csv)
        print(f"Data written to: {combined_csv}")