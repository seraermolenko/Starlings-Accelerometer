import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# NOTE: Description
# Takes a post-preprocessing parquet file with raw accelerometor data. 
# Performs an FFT for each second (100 data points per second) and then averages over two seconds. 
# The FFT transform and the averging over two seconds are broken down into seprate steps, allowing for adjustment if needed. 
# Outputs two CSV files, VeDBA and accZ. 

# NOTE:  Make adjustments as needed! 
# The section of the day that is kept can be adjusted at the bottom where it says start and end time, we kept daylight.
# The filtering of frequency values kept, inside of analyze intervals can also be adjusted.

# NOTE: Example run: python3 fft.py ../Starling

# NOTE: May take a few minutes for bigger files

# NOTE: If you only need accZ, or VeDBA, comment out one or the other!


inputs = sys.argv[1:]
SAMPLE_RATE = 100

# For get average function
chunk_length = 2 

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
    
    filtered_data = by_day

    interval_groups = filtered_data.resample('1s') 
    all_data_VeDBA = []
    all_data_accZ = []

    for interval, interval_data in interval_groups: 
        interval_data = interval_data.copy()

        interval_data['VeDBA'] = np.sqrt(interval_data['accX']**2 + interval_data['accY']**2 + interval_data['accZ']**2)

        for dimension in ['VeDBA','accX']:
            fft_result, freqs = fourier(interval_data[dimension], SAMPLE_RATE)

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

            fft_df['Datetime'] = interval

            if dimension == 'VeDBA':
                all_data_VeDBA.append(fft_df)
            else:
                all_data_accZ.append(fft_df)

    result_df_VeDBA = pd.concat(all_data_VeDBA, ignore_index=True)
    result_df_accZ = pd.concat(all_data_accZ, ignore_index=True)

    print("VeDBA DF before averaging per 2 seconds")
    print(result_df_VeDBA)
    print("accZ DF before averaging per 2 seconds")
    print(result_df_accZ)

    return result_df_VeDBA, result_df_accZ


# Averages the FFT data every 2 seconds. The get_avg can be changed if needed. 
def get_avg(fft_df, chunk_length):

    fft_df['Chunk'] = fft_df['Datetime'].dt.floor(f'{chunk_length}s')
    
    grouped_df = fft_df.groupby('Chunk').agg({'Frequency': 'mean', 'Magnitude': 'mean'}).reset_index()
    grouped_df.rename(columns={'Chunk': 'Datetime'}, inplace=True)
    grouped_df['Frequency'] = grouped_df['Frequency'].round(3)
    grouped_df['Magnitude'] = grouped_df['Magnitude'].round(3)

    print("result after grouping/averaging") 
    print(grouped_df)
    return grouped_df


for file_path in inputs:
    files = []
    if file_path.endswith('.parquet'):
        print(f'Opening file: {file_path}')
        data = pd.read_parquet(file_path)

        data = data.set_index('Datetime')    

        unique_days = data.index.date
        unique_days = pd.unique(unique_days)

        for day in unique_days:
            print(f'Processing day: {day}')
            
            # Define start and end times for the day
            start = f"{day} 06:00:00.000"  # Example: Start at 6 AM or "NONE"
            end = f"{day} 21:00:00.000"   # Example: End at 9 PM or "NONE"
            
            # Perform FFT analysis on the intervals
            result_df_VeDBA, result_df_accZ = analyze_intervals(data, day, start=start, end=end)
            
            if result_df_VeDBA.empty and result_df_accZ.empty :
                print(f"No data for {day} in the given interval.")
                continue
            
            # Averages the results over 2 seconds intervals 
            fft_df_VeDBA = get_avg(result_df_VeDBA, chunk_length)
            fft_df_accZ = get_avg(result_df_accZ, chunk_length)
            
            # Save results for VeDBA
            fft_df_VeDBA.index.name = 'Index'
            output_file_VeDBA = f'fft_VeDBA_results/{day}_fft_VeDBA.csv'
            fft_df_VeDBA.to_csv(output_file_VeDBA)
            print(f'Saved results for {day} (VeDBA) to {output_file_VeDBA}')

            # Save results for accZ
            fft_df_accZ.index.name = 'Index'
            output_file_accZ = f'fft_accZ_results/{day}_fft_accZ.csv'
            fft_df_accZ.to_csv(output_file_accZ)
            print(f'Saved results for {day} (accZ) to {output_file_accZ}')                                                                                                     
