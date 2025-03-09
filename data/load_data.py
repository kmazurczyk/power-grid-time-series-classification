
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import re
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from signal_processing import Signal, make_waves_for_df
from __init__ import get_base_path
from scipy.io import arff

load_dotenv()
raw_data_path = get_base_path() + os.getenv('RAW_DATA_DIR')
clean_data_path = get_base_path() + os.getenv('CLEAN_DATA_DIR')
label_file = raw_data_path + os.getenv('LABEL_FILE')
marker_file = raw_data_path + os.getenv('MARKER_FILE')
raw_csv = clean_data_path + os.getenv('RAW_DATA_FILE')
combined_csv = clean_data_path + os.getenv('COMBINED_DATA_FILE')
combined_sample_csv = clean_data_path + os.getenv('COMBINED_DATA_SAMPLE_FILE')

class DataLoader:

    def __init__(self):
        pass

    def __str__(self):
        print("DataLoader", dir(DataLoader), "configure file paths with .env")

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
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

        #---  FEATURES / COLUMNS ---#
        self.id_cols = ['source_file','sample_id']

        self.status_cols = [
            'control_panel_log1','control_panel_log2','control_panel_log3','control_panel_log4', \
            'relay1_log', 'relay2_log', 'relay3_log', 'relay4_log', \
            'snort_log1', 'snort_log2', 'snort_log3', 'snort_log4', \
            'R1_status_flag_for_relays','R2_status_flag_for_relays','R3_status_flag_for_relays','R4_status_flag_for_relays']

        self.target_features = ['is_attack','scenario_class','marker','scenario_type']

        self.R1_features = [i for i in self.columns if 'R1' in i]
        self.R2_features = [i for i in self.columns if 'R2' in i]
        self.R3_features = [i for i in self.columns if 'R3' in i]
        self.R4_features = [i for i in self.columns if 'R4' in i]
        self.magnitudes  = [i for i in self.columns if 'magnitude' in i.lower()]
        self.angles      = [i for i in self.columns if 'angle' in i.lower()]
        self.frequencies = [i for i in self.columns if 'freq' in i.lower()]
        self.impedance   = [i for i in self.columns if 'impedance' in i.lower()]
        self.pos_neg_zero = [i for i in self.columns if 'zero' in i.lower()]
        self.control_panels = [i for i in self.columns if 'control' in i.lower()]
        self.snort       = [i for i in self.columns if 'snort' in i.lower()]
        self.status_flags = [i for i in self.columns if 'status_flag' in i.lower()]
        self.waves       = [i for i in self.columns if '_wave' in i.lower()]


    def __str__(self):
        return f"DataPreprocessing Class: Contains dataframe: shape {self.data.shape}, columns {self.columns}, several preprocessing and feature engineering methods. Each method returns self for method chaining. See dir()."
    
    # --- DATA CLEANSING --- #
    def cast_data_types(self):
        """takes care pandas datatyping"""
        # fix numeric typing
        relays = ['R1','R2','R3','R4']
        for r in relays:
            self.data[f'{r}_appearance_impedance_for_relays'] = self.data[f'{r}_appearance_impedance_for_relays'].astype('float64')

        # fix byte string on marker class
        self.data['marker'] = self.data['marker'].astype('int').astype('category')

        # cast datetime str to datetime if it's already in the dataset, pass if it's not
        try:
            self.data['synthetic_datetime'] = pd.to_datetime(self.data['synthetic_datetime'])
        except KeyError:
            pass 

        return self
    
    def apply_event_labels(self, marker_file: str, key: str):
        markers = pd.read_csv(marker_file)
        self.data = self.data.merge(right=markers, how='left', on=key)
        return self
        
    def impute_value_ffill(self, cols: list, value = None):
        """for df or series - returns pd.ffill for either NaN or a value (float) that >= will be nullified and imputed (float)"""
        if cols is None:
            cols = self.data.columns
        if value is not None:
            imputed_df = self.data[cols].map(lambda x: np.nan if x >= value else x)
        self.data[cols] = imputed_df.ffill(axis=0)
        return self

    # --- TRANSFORMERS --- #
    def std_scale_transform(self, cols: list, fit=True):
        scaled_cols = self.data.loc[:,cols]
        if fit:
            self.scaler.fit(scaled_cols)
        self.data[cols] = self.scaler.transform(scaled_cols)
        return self
    
    def one_hot_transform(self, cols: list, fit=True):
        # useful for log/status flags
        encoded_cols = self.data.loc[:,cols]
        if fit:
            encoded_cols = self.one_hot_encoder.fit(encoded_cols)
        new_cols = self.one_hot_encoder.get_feature_names_out()
        return new_cols, self.one_hot_encoder.transform(encoded_cols)

    def log_scale_transform(self, cols: list):
        # useful for pre-processing magnitude columns, which have high variance and >= 0
        # log_x+1 so useful for display but not statistical
        scaled_cols = self.data.loc[:,cols]
        self.data[cols] = scaled_cols.apply(lambda x: np.log1p(x), axis=1)
        return self

    # --- FEATURE ENGINEERING --- #   
    def binarize_classes(self, feature_name: str, new_feature_name: str, criteria: list):
        """for series: reduces multi class labels to binary. labels 1 for class labels found in list else 0"""
        criteria = [str.lower(i) for i in criteria]
        self.data[new_feature_name] = self.data[feature_name].apply(lambda x: 1 if x.lower() in criteria else 0)
        return self
    
    def binarize_values(self, cols: list, value=np.inf):
        """for df or series - generates binary 0/1 for >= continuous value, for known np.inf values on impedance"""
        value_cols = self.data[cols].map(lambda x: 1 if x >= value else 0)
        self.data[cols] = value_cols
        return self

    def binarize_status_flags(self, cols: list):
        """for df or series - reduces several discrete codes found under _status_flag_for_relays - if 0 then 0 else 1"""
        self.data[cols] = self.data[cols].map(lambda x: 0 if x==0 else 1)
        return self

    def generate_sample_ids(self, unique_columns=['marker','source_file']):
        """if the class marker or file number changes, consider it the start of a new sample and increment id"""
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
        # offset = timedelta(seconds=min_increment)   # offset between samples

        # partition the samples
        sample_id_counts = self.data['sample_id'].value_counts().to_dict()

        # increment time
        for s, n in sample_id_counts.items():
            # timestamp += offset                     # offset isn't working
            for i in range(n):
                timestamp += delta
                synthetic_datetime += [timestamp]

        self.data['synthetic_datetime'] = synthetic_datetime
        self.data['synthetic_datetime'] = pd.to_datetime(self.data['synthetic_datetime'])
        return self
    
    def sample_data(self, n_samples = None, replace=False) -> pd.DataFrame:
        # full data is about 80MB, 220 after feature engineering, ~80K rows. this returns a smaller sample of the data.
        # randomly samples without replacement from bag of sample_ids
        # default n is 1/3 of sample_ids
        if n_samples == None:
            n_samples = self.data['sample_id'].nunique() // 3
        sample = np.random.choice(self.data['sample_id'].unique(), size=n_samples, replace=replace)
        self.data = self.data.loc[self.data['sample_id'].isin(sample),:]
        return self, sample

    def get_dataframe(self) -> pd.DataFrame:
        return self.data
    
    def preprocess_pipeline(self):
        self.cast_data_types() \
            .apply_event_labels(marker_file, 'marker') \
            .binarize_status_flags([i for i in dp.columns if 'status_flag_for_relays' in i]) \
            .binarize_values(self.impedance) \
            .log_scale_transform(self.magnitudes) \
            .generate_sample_ids() \
            .generate_time_series()
            # .binarize_classes('scenario_class', 'is_attack', ['attack']) \

        return self

def load_from_csv(csv_file):      
    if os.path.exists(csv_file):
        pass
    else:
        load_data.__main__()
    df = pd.read_csv(csv_file,index_col=0)

    # data typing that is not retained by CSV
    dp = DataPreprocessor(df)
    df = dp.cast_data_types().get_dataframe()

    print(df.shape)
    return df

if __name__ == '__main__':

    # LOAD DATA
    dl = DataLoader()
    df = dl.load_arff_data(raw_data_path)
    df = dl.apply_column_names(label_file, df)
    df.to_csv(raw_csv)
    print(f"Raw data written to: {combined_csv}")

    # CLEAN DATA, FEATURE ENGINEERING
    dp = DataPreprocessor(df)
    df = dp.preprocess_pipeline().get_dataframe()
    df = make_waves_for_df(df)

    # drop sample with line shift on R4 values, indexes 19424:19431
    s15936 = df.loc[df['sample_id']==15936,:]
    df.drop(index=s15936.index, inplace=True)

    df.to_csv(combined_csv)
    print(f"Pre-processed data written to: {combined_csv}")

    # DEV SAMPLE
    dp = DataPreprocessor(df)
    dp.sample_data()
    df = dp.get_dataframe()
    df.to_csv(combined_sample_csv)
    print(f"Dev sample data written to: {combined_sample_csv}")
