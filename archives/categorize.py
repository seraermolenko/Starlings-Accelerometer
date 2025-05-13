import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

# OVERVIEW:
# Seperating the Behvioral Classificication/Decision tree work 
# Includes an avg over data and more percises decsision tree

# TO RUN:
# python3 categorize.py ../Starling/EUST_001_BOX260_MAY-2.parquet


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
    

    print("Filtered DF")
    print(filtered_data)
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
        
            # Only keep the positive freq for now
            fft_df = fft_df[fft_df['Frequency'] > 0]

            # Freq under 30 
            fft_df = fft_df[fft_df['Frequency'] <= 30]

            fft_df = fft_df[fft_df['Magnitude'] > 0.5]

            all_data.append(fft_df)

    result_df = pd.concat(all_data, ignore_index=True)
    print("result DF")
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

def apply_decision_tree(fft_df):
    behaviors = []
    current_duration = 0
    previous_behavior = None

    for _, row in fft_df.iterrows():
        magnitude = row['Magnitude']
        frequency = row['Frequency']

        if magnitude < 0.5:
            behavior = 'Idle'  

        elif 10 <= frequency <= 20:
            if previous_behavior == 'Ruffle':
                current_duration += 2
            else: 
                current_duration = 2

            if current_duration >= 2:
                behavior = 'Flying'
            else: 
                behavior = 'Ruffle'

        elif 0.5 <= frequency <= 4:
            if previous_behavior == 'Feeding':
                current_duration += 2  
            else:
                current_duration = 2

            if current_duration <= 8:
                behavior = 'Feeding'
            else:
                behavior = 'Foraging'

        else:
            behavior = 'Not Flying'

        behaviors.append({
            'Frequency': frequency,
            'Magnitude': magnitude,
            'Behavior': behavior
        })

        previous_behavior = behavior 

    behavior_df = pd.DataFrame(behaviors)
    print(behavior_df)

    return behavior_df

def visualize_behaviors(fft_df):
    plt.figure(figsize=(10, 6))

    # Color Coded for Behaviours
    sns.scatterplot(
        data=fft_df, 
        x='Frequency', 
        y='Magnitude', 
        hue='Behavior', 
        palette='viridis', 
        s=50
    )

    plt.title('VDBA-Daytime-2s Avg:Behavior Clustering Based on FFT Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend(title='Behavior')
    plt.savefig('VDBA-DayTime-Behavior-2avg-Clustering_2020-05-02.png')

def fft_plot(fft_df):
    plt.figure(figsize=(10,5))
    plt.plot(fft_df.index, fft_df['Magnitude'])
    plt.title("Magnitude vs. Time")
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.savefig('visualize.png')

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

        # Decision Tree
        behavior_df = apply_decision_tree(fft_df)

        # Graphs decision tree df
        visualize_behaviors(behavior_df)

        start_time = 0
        end_time = 4 * 60

        fft_df.index.name = 'Time'
        fft_df['Magnitude'] = fft_df['Magnitude'].round(3)

        fft_df_chunk = fft_df[(fft_df.index >= start_time) & (fft_df.index < end_time)]

        #fft_plot(fft_df_chunk)

