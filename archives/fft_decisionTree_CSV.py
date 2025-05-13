import pandas as pd
import numpy as np
import sys


# OVERVIEW:
# Seperating the creation of a file from fft results for ease of use 

# TO RUN:
# python3 fft_decisionTree_CSV.py ../Starling/EUST_001_BOX260_MAY-2.parquet


inputs = sys.argv[1:]
SAMPLE_RATE = 100

def fourier(data, sample_rate):
    fourier = np.fft.fft(data)
    magnitude = np.abs(fourier) / (sample_rate / 2)              # Normalizing magnitude
    freqs = np.fft.fftfreq(len(data), 1 / sample_rate)
    return magnitude, freqs

def analyze_intervals(data, day, start=None, end=None):
    by_day = data[data.index.date == pd.to_datetime(day).date()]  # Contains data for one day
    print(f"Data available for {day}: {by_day.shape}")

    if start and end:
        filtered_data = by_day[(by_day.index >= start) & (by_day.index <= end)]
    else:
        filtered_data = by_day
    

    #print("Filtered DF")
    #print(filtered_data)
    filtered_data = by_day
    all_data = []


    interval_groups = filtered_data.resample('1s') 
    #print(list(interval_groups))

    for interval, interval_data in interval_groups: 
        interval_data = interval_data.copy()

        # Calculate the mean of the three axes
        #interval_data.loc[:, 'mean'] = interval_data[['accX', 'accY', 'accZ']].mean(axis=1) 

        interval_data['VeDBA'] = np.sqrt(interval_data['accX']**2 + interval_data['accY']**2 + interval_data['accZ']**2)

        interval_start = interval
        interval_end = interval  + pd.Timedelta('3min')

        # Analyze and plot FFT for each dimension       #['accX', 'accY', 'accZ', 'mean']:
        for dimension in ['VeDBA']:
            fft_result, freqs = fourier(interval_data[dimension], SAMPLE_RATE)

            # COMMENT OUT HERE (FFT PLOTS)
            #plot_fft_mag_over_freq(interval_start, interval_end, day, dimension, fft_result, freqs)                   
            #plot_freq_over_time(interval_start, interval_end, day, dimension, fft_result, freqs)                      

            fft_df = pd.DataFrame({
                'dimension': dimension,
                'Frequency': freqs,
                'Magnitude': np.abs(fft_result)
            })
        
            # Keeing positive frequency only
            fft_df = fft_df[fft_df['Frequency'] > 0]

            # Keeping frequency 30 and under
            fft_df = fft_df[fft_df['Frequency'] <= 30]

            # Keeping Magnitude grater than 0.5
            fft_df = fft_df[fft_df['Magnitude'] > 0.5]

            fft_df['datetime'] = interval
            all_data.append(fft_df)

    result_df = pd.concat(all_data, ignore_index=True)
    print("DF before averging per 2 seconds")
    print(result_df)

    return result_df

def get_avg(fft_df):

    if len(fft_df) % 2 != 0:              
        fft_df = fft_df[:-1]

    fft_df = fft_df.reset_index(drop=True)
    avg_freq = []
    avg_mag = []
    time = []

    for i in range(0,len(fft_df), 2):
        avg_freq.append(np.mean(fft_df['Frequency'].iloc[i:i+2]))
        avg_mag.append(np.mean(fft_df['Magnitude'].iloc[i:i+2]))
        time.append(i // 2 * 2)

    avg_df = pd.DataFrame({
            'Frequency': avg_freq,
            'Magnitude': avg_mag
        })
    avg_df.index = time

    # print("Avg df")     
    # print(avg_df)
    return avg_df


for file_path in inputs:
    files = []
    if file_path.endswith('.parquet'):
        print(f'Opening file: {file_path}')
        data = pd.read_parquet(file_path)

        # PICK A DAY HERE
        #print(data['Datetime'].dt.date.value_counts())                                 # Lets me know what days, and the values per day
        
        #data = data['Datetime'] = pd.to_datetime(data['Datetime'])                     # Converting to dattime (bug error for resample)
        data = data.set_index('Datetime')                                               # Using timestamp col as index for plotting (plotitng over time)

        # COMMENT OUT HERE (PLOTTING)
        #plotData(data)                                                                  

        # INPUT HERE, OR "NONE"
        day = "2020-05-02"
        start= "2020-05-02 6:00:00.000"                                                 
        end= "2020-05-02 21:00:00.000"
    
        # FFT on intervals å
        fft_df = analyze_intervals(data, day, start=start, end=end)

        # Avergaes the frequencies post FFT's on 1s intervals 
        fft_df = get_avg(fft_df)

        fft_df.index.name = 'Time'
        fft_df['Magnitude'] = fft_df['Magnitude'].round(3)

        fft_df.to_csv(f'fft_VeDBA_results/{day}_fft_VeDBA_results.csv')

