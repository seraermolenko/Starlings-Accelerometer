import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import seaborn as sns


#NOTE: Example run: python3 decision_tree.py ../starlings-accelerometer/fft_VeDBA_results/2020-05-03_fft_VeDBA.csv 

#NOTE: The duration and freq thresholds, as well as the attributions for them have been set semi-randomly by non-biologists and are meant to be adjusted!

#NOTE: Should output a scatterplot and step plot into decision_tree_results folder

#NOTE: This script applies a decision tree on the entire csv file input, no start or end point. 

#NOTE: Width of step plot, the higher the number the more zoomed in the plot will be. 
ZOOM = 50   


inputs = sys.argv[1:]
file_path = inputs[0]
fft_df = pd.read_csv(file_path)
fft_df['Datetime'] = pd.to_datetime(fft_df['Datetime'])

def fft_plot(fft_df):
    plt.figure(figsize=(ZOOM,5))
    plt.step(fft_df['Datetime'], fft_df['Magnitude'], where='post')
    date_str = fft_df['Datetime'].iloc[0].strftime('%Y-%m-%d')
    output_file = f'decision_tree_results/{date_str}visualize.png'
    plt.title(f'{date_str} Magnitude vs. Time')
    plt.xlabel('Datetime')
    plt.ylabel('Magnitude')
    plt.gcf().autofmt_xdate() 
    plt.savefig(output_file)
    print(f'Saved plotting results {output_file}')

def apply_decision_tree(fft_df):
    behaviors = []
    current_duration = 0
    previous_behavior = None

    for _, row in fft_df.iterrows():
        magnitude = row['Magnitude']
        frequency = row['Frequency']
        datetime = row['Datetime']

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
            'Datetime': datetime,
            'Frequency': frequency,
            'Magnitude': magnitude,
            'Behavior': behavior
        })

        previous_behavior = behavior 

    behavior_df = pd.DataFrame(behaviors)
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

    date_str = fft_df['Datetime'].iloc[0].strftime('%Y-%m-%d')
    output_file = f'decision_tree_results/{date_str}_Behavior_Clustering.png'
    plt.title(f'{date_str} Behavior Clustering Based on FFT Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend(title='Behavior')
    plt.savefig(output_file)
    print(f'Saved decision tree results {output_file}')

fft_plot(fft_df)
behavior_df = apply_decision_tree(fft_df)
visualize_behaviors(behavior_df)


         
