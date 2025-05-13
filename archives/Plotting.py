import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Plotting data 
# Forier Transofrmations
# python3 Plotting.py ../Starling/EUST_001_BOX260_MAY-2.parquet

#EUST_008_BOX305_APR-30-2021
# EUST_001_BOX260_MAY-2.parquet

# Datetime
# 2020-05-03    8640000
# 2020-05-04    8640000
# 2020-05-05    4169401
# 2020-05-02    4029400

inputs = sys.argv[1:]
SAMPLE_RATE = 100

def plotData(data):

    # ADJUST DAYS HERE, CAN USE PICK A DAY TO OUTPUT YOUR DAYS 
    days = ['2020-05-03','2020-05-04','2020-05-05','2020-05-02']
    for day in days:
        by_day = data[data.index.date == pd.to_datetime(day).date()]                    # Comparing only the data part of datetime 
        
        plt.figure(figsize=(40,10))

        # COMMENT OUT HERE (PICK A DIMENSION)
        plt.plot(by_day.index, by_day['accX'])
        # plt.plot(by_day.index, by_day['accY'])
        # plt.plot(by_day.index, by_day['accZ'])
        # by_day['mean'] =  by_day[['accX', 'accY', 'accZ']].mean()
        # plt.plot(by_day.index, by_day['mean'])

        # Remember to change 'X' 'Y' 'Z' 'mean   
        plt.title(f'X Accelerometer Data Day {day}')                                 
        plt.xlabel('Date')
        plt.ylabel('Acceleration')
        plt.legend()

        # Remember to change 'X' 'Y' 'Z' 'mean' (or code smth better later lol)
        plt.savefig(f'X_{day}.png')                                                     

def fourier(data, sample_rate):
    fourier = np.fft.fft(data)
    magnitude = np.abs(fourier) / (sample_rate / 2)              # Normalizing magnitude
    freqs = np.fft.fftfreq(len(data), 1 / sample_rate)
    return magnitude, freqs

def plot_fft_mag_over_freq(start_time, end_time, day, dimension, fft_result, freqs):
    plt.figure(figsize=(20, 10))
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(fft_result)//2])                            # Plotting positive freq only
    plt.title(f'FFT of {dimension} on {day} from {start_time} to {end_time}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.savefig(f'Mag_over_freq_{day}_{dimension}_{start_time}:{end_time}.png')

def plot_freq_over_time(start_time, end_time, day, dimension, fft_result, freqs):
    plt.figure(figsize=(20, 10))
    pos_mask = freqs > 0  # Podititve freq only
    plt.plot(freqs[pos_mask], np.abs(fft_result[pos_mask]), label='Magnitude')

    plt.title(f'Frequency vs Time for {dimension} on {day} from {start_time} to {end_time}')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.grid()
    plt.legend()
    plt.savefig(f'Frequency_Over_Time_{day}_{dimension}_{start_time}:{end_time}.png')

def analyze_intervals(data, day, start=None, end=None):
    by_day = data[data.index.date == pd.to_datetime(day).date()]  # Contains data for one day
    print(f"Data available for {day}: {by_day.shape}")

    # if start and end:
    #     filtered_data = by_day[(by_day.index >= start) & (by_day.index <= end)]
    # else:
    #     filtered_data = by_day
    

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

def apply_kmeans(data):

    #fft_df['Time'] = fft_df.index  
    X = data[['Frequency', 'Magnitude']] 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # scatter = ax.scatter(
    #     data['Frequency'], 
    #     data['Magnitude'], 
    #     data.index,  # Use index or another variable for Z-axis if needed
    #     c=data['Cluster'], 
    #     cmap='viridis', 
    #     s=50
    # )

    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Magnitude')
    # ax.set_zlabel('Time Index (or Dimension)')
    # plt.title('3D Scatter Plot of FFT Data with Clusters')
    #plt.show()

    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(data=data, x='Frequency', y='Magnitude', hue='Cluster', palette='viridis', s=50)
    #sns.scatterplot(data=data, x='Time', y='Frequency', hue='Cluster', palette='viridis', s=50)

    # plt.title('Frequency vs Time by Cluster')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency (Hz)')

    plt.title('Behavior, K-means Clustering of FFT Data')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.savefig('Behavior-Clustering-2020-05-01')

    return 

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
                current_duration += 1
            else: 
                current_duration = 1

            if current_duration >= 2:
                behavior = 'Flying'
            else: 
                behavior = 'Ruffle'

        elif 0.5 <= frequency <= 3:
            if previous_behavior == 'Feeding':
                current_duration += 1  
            else:
                current_duration = 1 

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


    plt.title('VDBA-Daytime Behavior Clustering Based on FFT Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend(title='Behavior')
    plt.savefig('Time-VDBA-DayTime-Behavior-Clustering_2020-05-02.png')


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

    
        # COMMENT OUT HERE (FFT) (KNEANS)
        fft_df = analyze_intervals(data, day, start=start, end=end)
        #apply_kmeans(fft_df)

        behavior_df = apply_decision_tree(fft_df)
        #apply_kmeans(behavior_df)

        visualize_behaviors(behavior_df)



# From Meghans 
# "2020-05-01"
# "2020-05-01 10:38:30.000"
# "2020-05-01 10:34:40.000"