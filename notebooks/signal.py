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
        self.freq = 60
        self.sample_rate = 120
        self.timespan = self.data.shape[0]

    def __str__(self):
        pass

    def degree_to_radian(self, theta) -> pd.Series:
        return theta * np.pi / 180
    
    def log_scale_transform(self, cols: list):
        # useful for pre-processing magnitude columns, which have high variance and >= 0
        # log_x+1 so useful for display but not statistical
        scaled_cols = self.data.loc[:,cols]
        self.data[cols] = scaled_cols.apply(lambda x: np.log(x+1), axis=1)
        return self

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

    def make_waves_iter(self, relays: list, phases: list, stats=['voltage','current','power'], zero=True) -> pd.DataFrame:
        # easy way to dynamically calc/plot waveforms for relay, phase, stat combinations
        # modifies self.data and also returns columns if subsetting

        df = pd.DataFrame()
        for i in relays: 
            for j in phases:
                # V
                if 'voltage' in stats:
                    v_mag, v_angle = self.data[f'{i}_voltage_AC_phase_{j}_magnitude'], self.data[f'{i}_voltage_AC_phase_{j}_angle']
                    n0_v_mag, n0_v_angle = self.data['R1_pos_neg_zero_voltage_phase_A_magnitude'],self.data['R1_pos_neg_zero_voltage_phase_A_angle']                    
                    v_wave, n0_v_wave = self.waveform(v_mag, v_angle), self.waveform(n0_v_mag, n0_v_angle)
                    
                    df[f'{i}_Phase_{j}_voltage'] = v_wave
                    df[f'{i}_pos_neg_zero_phase_voltage_phase_{j}']= n0_v_wave
                    
                # I 
                if 'current' in stats:
                    i_mag, i_angle = self.data[f'{i}_current_AC_phase_{j}_magnitude'], self.data[f'{i}_current_AC_phase_{j}_angle']
                    n0_i_mag, n0_i_angle = self.data['R1_pos_neg_zero_current_phase_A_magnitude'],self.data['R1_pos_neg_zero_current_phase_A_angle']
                    i_wave, n0_i_wave = self.waveform(i_mag, i_angle), self.waveform(n0_i_mag, n0_i_angle)
                    
                    df[f'{i}_Phase_{j}_current'] = i_wave
                    df[f'{i}_pos_neg_zero_phase_current_phase_{j}']= n0_i_wave
              
                # P
                if 'power' in stats:
                    try:
                        p_wave, p_avg = self.instant_power(v_wave, i_wave), self.avg_power(v_mag, i_mag, v_angle, i_angle)
                        n0_p_wave, n0_p_avg = self.instant_power(n0_v_wave, n0_i_wave), self.avg_power(v_mag, i_mag, v_angle, i_angle)

                        df[f'{i}_Phase_{j}_power'] = p_wave
                        df[f'{i}_Phase_{j}_avg_power'] = p_avg
                        df[f'{i}_pos_neg_zero_phase_{j}_power'] = n0_p_wave
                        df[f'{i}_pos_neg_zero_phase_{j}_avg_power'] = n0_p_avg

                    except Exception as e:
                        print(e)
                        print('You can only calc power if you also selected voltage and current.')

                # Positive Negative Zero Series
                if zero == False:
                    zero_cols = [i for i in df.columns if 'zero' in i.lower()]
                    df.drop(columns=zero_cols, inplace=True)

        self.data = pd.concat([self.data, df], axis=1)
        return df
    
if __name__ == '__main__':
    pass