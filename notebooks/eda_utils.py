from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from __init__ import get_base_path

load_dotenv()
clean_data_path = get_base_path() + os.getenv('CLEAN_DATA_DIR')
combined_csv = clean_data_path + os.getenv('COMBINED_DATA_FILE')

class Eda:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.columns = data.columns.to_list()
        self.summary_df_stats = self.get_summary_df_stats()
        self.summary_col_stats = self.get_summary_col_stats()

    def __str__(self):
        print(f"EDA with Summary DF Statistics: {self.summary_df_stats.columns.to_list()} \n Summary Column Statistics: {self.summary_col_stats.columns.to_list()}")
        
    def get_summary_df_stats(self):
        stats_dict = {
            'n_rows': self.data.shape[0],
            'n_cols': self.data.shape[1],
            'n_nulls': self.data.isna().sum().sum()
        }
        return pd.DataFrame.from_dict(stats_dict,orient='index').T

    def get_summary_col_stats(self):
        numeric_describe = self.data.describe().round(2).T
        
        stats_dict = {
            'dtype': {i: self.data[i].dtype for i in self.columns},
            'pct_null': {i: self.data[i].isnull().sum()/len(self.data[i]) for i in self.columns},
            'n_unique': {i: self.data[i].nunique() for i in self.columns}
        }
        stats_df = pd.DataFrame.from_dict(stats_dict)
        stats_df = stats_df.merge(right=numeric_describe, how='left', left_index=True, right_index=True)

        return stats_df

    def get_class_counts(df, class_name, index=0):
        '''
        input: 
            dataframe
            optionally an index/row number can be passed
        counts rows per class_name (using the marker column)
        returns: dataframe
        '''

        stats_dict = df.loc[:,[class_name,'marker']].groupby(class_name).count().T.to_dict(orient='records')
        stats_df = pd.DataFrame(stats_dict, index=[index])
        return stats_df

class Signal:
    # utilities for breaking apart and visualizing key AC components
    # frequency is indicative from the data, sample rate is from data owner

    def __init__(self, data):
        self.data = data
        self.freq = 60
        self.sample_rate = 120
        self.timespan = self.data.shape[0]

    def __str__(self):
        pass

    def degree_to_radian(self, theta) -> pd.Series:
        return theta * np.pi / 180

    def time_distance(self) -> pd.Series:
        time = self.timespan / self.freq * self.sample_rate
        return np.linspace(0, time, self.timespan)

    def waveform(self, RMS_amp, theta) -> pd.Series:
        # valid for voltage and current. expects theta in degrees
        # synchrophasor magnitude is typically given as RMS value, so we must convert
        max_amp = RMS_amp * np.sqrt(2) 
        theta = self.degree_to_radian(theta)
        time = self.time_distance()
        return max_amp * np.cos(2*np.pi*self.freq*time + theta)
    
    def instant_power(self, voltage_wave, current_wave)-> pd.Series:
        return voltage_wave * current_wave

    def avg_power(self, voltage_RMS_mag, current_RMS_mag, voltage_angle, current_angle) -> pd.Series:
        # expects theta in degrees
        voltage_angle, current_angle = [self.degree_to_radian(i) for i in (voltage_angle, current_angle)]
        power_factor = np.cos(voltage_angle - current_angle)
        return (voltage_RMS_mag * current_RMS_mag / 2) * power_factor

    # if we add impedance r
    # i * r = v
    # i^2 * r = w
    # i * v = w

if __name__ == '__main__':
    df = pd.read_csv(combined_csv, index_col=0)
    eda = Eda(df)
