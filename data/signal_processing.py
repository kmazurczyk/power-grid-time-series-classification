from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from __init__ import get_base_path

class Signal:
    # utilities for breaking apart and visualizing key AC components
    # frequency is indicative from the data, sample rate is from data owner

    def __init__(self, data):
        self.data = data
        self.columns = self.data.columns
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
        # valid for voltage and current. expects theta in degrees and converts to radians
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

    def add_noise(self, wave: pd.Series) -> pd.Series:
        noise = []
        noisy_wave = wave + noise
        pass

    def mean_wave_smoothing(self, wave: pd.Series, window: int) -> pd.Series: 
        # ideally include padding to avoid edge effects
        pass
    
    def make_waves_iter(self, relays=('R1','R2','R3','R4'), phases=('A','B','C'), stats=['voltage','current','power'], zero=True) -> pd.DataFrame:
        # easy way to dynamically calc/plot waveforms for relay, phase, stat combinations
        # modifies self.data and also returns columns if subsetting

        df = pd.DataFrame()
        
        for i in relays: 
            for j in phases:
                # V
                if 'voltage' in stats:
                    v_mag = self.data[f'{i}_voltage_AC_phase_{j}_magnitude']
                    v_angle = self.data[f'{i}_voltage_AC_phase_{j}_angle']
                    n0_v_mag, n0_v_angle = self.data[f'{i}_pos_neg_zero_voltage_phase_{j}_magnitude'], self.data[f'{i}_pos_neg_zero_voltage_phase_{j}_angle']                    
                    v_wave, n0_v_wave = self.waveform(v_mag, v_angle), self.waveform(n0_v_mag, n0_v_angle)
                    
                    df[f'{i}_Phase_{j}_voltage_wave'] = v_wave
                    df[f'{i}_Phase_{j}_pos_neg_zero_voltage_wave']= n0_v_wave
                    
                # I 
                if 'current' in stats:
                    i_mag, i_angle = self.data[f'{i}_current_AC_phase_{j}_magnitude'], self.data[f'{i}_current_AC_phase_{j}_angle']
                    n0_i_mag, n0_i_angle = self.data[f'{i}_pos_neg_zero_current_phase_{j}_magnitude'],self.data[f'{i}_pos_neg_zero_current_phase_{j}_angle']
                    i_wave, n0_i_wave = self.waveform(i_mag, i_angle), self.waveform(n0_i_mag, n0_i_angle)
                    
                    df[f'{i}_Phase_{j}_current_wave'] = i_wave
                    df[f'{i}_Phase_{j}_pos_neg_zero_current_wave']= n0_i_wave
              
                # P
                if 'power' in stats:
                    try:
                        p_wave, p_avg = self.instant_power(v_wave, i_wave), self.avg_power(v_mag, i_mag, v_angle, i_angle)
                        n0_p_wave, n0_p_avg = self.instant_power(n0_v_wave, n0_i_wave), self.avg_power(v_mag, i_mag, v_angle, i_angle)

                        df[f'{i}_Phase_{j}_power_wave'] = p_wave
                        df[f'{i}_Phase_{j}_avg_power'] = p_avg
                        df[f'{i}_Phase_{j}_pos_neg_zero_power_wave'] = n0_p_wave
                        df[f'{i}_Phase_{j}_pos_neg_zero_avg_power'] = n0_p_avg

                    except Exception as e:
                        print(e)
                        print('You can only calc power if you also selected voltage and current.')

                # Positive Negative Zero Series
                if zero == False:
                    zero_cols = [i for i in df.columns if 'zero' in i.lower()]
                    df.drop(columns=zero_cols, inplace=True)

        self.data = pd.concat([self.data, df], axis=1)

        # append datetime
        try:
            df['synthetic_datetime'] = self.data['synthetic_datetime']
        except KeyError as e:
            print('Add "synthetic_datetime" to your dataframe')
            print(e)

        return df

def make_waves_for_df(df):
    ''' takes dataframe
        partitions and computes signal by sample_id
        returns dataframe with new columns from Signal.make_waves_iter'''
    wave_df = pd.DataFrame()
    for sample_id in df['sample_id'].unique():
        s = Signal(df.loc[df['sample_id'] == sample_id, :])
        s.make_waves_iter(zero=True)
        wave_df = pd.concat([wave_df,s.data],axis=0)
    return wave_df

def describe_wave(wave: pd.Series):
    return {
        'mean': wave.mean(),
        'median': wave.median(),
        'q25': wave.quantile(0.25),
        'q75': wave.quantile(0.75),
        'sigma': wave.std(),
        'min': wave.min(),
        'max': wave.max(),
        'zero_crossing_rate': np.sum(np.diff(np.signbit(wave))) / len(wave) # proportion of times the signal crosses 0
    }

if __name__ == '__main__':
    pass