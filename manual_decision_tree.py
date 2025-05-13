import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Date formatting for graphs
import sys
import os
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN, AgglomerativeClustering, Birch, HDBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure

# Input: A folder containing transformed files to label
# Output: For each file, bird categories are assigned and added as 'behavior' dataframe column. All files are saved into provided output file
# NOTE: each file takes about 1 minute to save.

# Example command: 
# python manual_decision_tree.py "../input_folder" "../output_folder"

# 1. Global Parameters
GOOD_AMP = 0.5
MAX_FREQ = 30 
NUM_CLUSTERS = 2
ALPHA = 0.05 


def process(X):


    # filter out amplitudes less than GOOD_AMP threshold (usually 0.3 or 0.5)
    X = X[(X['ampX'] >= GOOD_AMP) & (X['ampY'] >= GOOD_AMP) & (X['ampZ'] >= GOOD_AMP) & (X['amp'] >= GOOD_AMP)] # only significant amplitude
    # filter out frequencies greater than MAX_FREQ threshold (usually 30Hz)
    X = X[(X['freqX'] <= MAX_FREQ) & (X['freqY'] <= MAX_FREQ) & (X['freqZ'] <= MAX_FREQ) & (X['freq'] <= MAX_FREQ)] # remove frequency above 30 Hz
    
    # Calculate absolute values of accX, accY, and accZ
    abs = X[['accX', 'accY', 'accZ']].abs()

    column_indices = {col: X.columns.get_loc(col) for col in ['accX', 'accY', 'accZ']}
    # Note: map accX, accY, accZ to numerical attributes

    # Add max acceleration feature
    X['max_accl'] = abs.max(axis=1)
    #X['max_dim'] = abs.idxmax(axis=1).map(column_indices)

    # Add min acceleration feature
    X['min_accl'] = abs.min(axis=1)
    #X['min_dim'] = abs.idxmin(axis=1).map(column_indices)

    X = X.drop(['acclist'], axis=1)
    #X = X.drop(columns = ['accX', 'accY', 'accZ', 'acc'])

    
    return X

## NOTE MODIFY AS NEEDED
def main():
    #infile = "../transformed_data_3fourier"
    infile = sys.argv[1]
    outfile = sys.argv[2]

    #errors if files provided as argv aren't valid
    if not os.path.isdir(sys.argv[1]):
        sys.exit(f"Input directory '{infile}' does not exist.")
    if not os.path.isdir(sys.argv[2]): # if output file doesn't exist, create it.
        os.makedirs(outfile)



    for file_path in Path(infile).iterdir():
        if file_path.is_file():
            try:

                X = pd.read_parquet(file_path, engine='pyarrow')
                file_name = Path(file_path).name # get filename
                print(f"Processing file: {file_name}")

                # For each file in input directory, generate dayplots
                generate_behavior_labels(X, file_name, outfile)

            except Exception as e:
                print(f"Skipping file {file_path.name}: {e}")


## NOTE MODIFY AS NEEDED
#fig_name = "figures/rachel_nov21/ex2_fourier3_ampZ" # for day
# X = pd.read_parquet(infile, engine='pyarrow')
# file_name = Path(infile).name # get filename

# print(X)

#exit()

## example 1
# start_time = "2020-05-01 10:34:40.000"
# end_time = "2020-05-01 10:38:30.000"

# example 2
# start_time = "2020-05-01 10:31:20.000"
# end_time = "2020-05-01 10:32:20.000"

# Showing 5 minutes of April 30th only # # filter for data points in this range
#X = X.loc[X.index.date == pd.to_datetime('2020-05-01').date()]
# X = X.loc[start_time:end_time]


# NOTE MODIFY AS NEEDED plotted x and y dimensions, change to freqY/ampY, freqZ/ampZ, freq/amp as needed # VDBA
def generate_behavior_labels(X, file_name, outfile):
    the_freq = 'freqZ' # X
    the_amp = 'ampZ' # Y, also magnitude

    ## NOTE MODIFY AS NEEDED
    val =  the_amp # value we're plotting by


    # Labels assigned according to Example 1 in EUST accelerometer signature analysis powerpoint (Megan's work)
    categories = [(8 <= X[the_freq]) & (X[the_amp] >= 0.8), # Green, Takeoff!
                    ((0 <= X[the_freq]) & (X[the_freq] <= 10) & (X[the_amp] >= 0.2)) | ((X[the_freq] <= 15) & (X[the_amp] <= 0.5) & (X[the_amp] >= 0.2)), # Black, Slow Flight/End of Flight
                    (10 < X[the_freq]) & (X[the_freq] < 15) & (X[the_amp] >= 0.5), # Red, Fast Flight/Start of Flight
                    (18 <= X[the_freq]), # Feather Ruffling
                    (X[the_amp] < 0.2) ] # Sitting
                    #(30 < X[the_freq]) & (X[the_freq] <= 50) & (X[the_amp] >= 0.5)] # Orange, Zooming
    remaining = ~np.any(categories, axis=0)

    categories.insert(0, remaining) # prepend so it's plotted first

    values = ['unknown', 'takeoff','slowflight', 'fastflight', 'ruffling', 'sitting']
    X['behavior'] = np.select(categories, values)

    colors = {'unknown': 'blue', 'slowflight':'black','fastflight':'red','takeoff':'green', 'ruffling':'orange', 'sitting':'magenta'}
    t = 20
    marker_thickness = {'unknown': 1, 'slowflight': t,'fastflight': t,'takeoff': t, 'ruffling': t, 'sitting': t}
    X['color'] = X['behavior'].apply(lambda x: colors[x])
    X['thickness'] = X['behavior'].apply(lambda x: marker_thickness[x])


    # drop color and thickness columns, used for plot visualizing
    X.drop(['color', 'thickness'], axis=1)

    # Save newly generated labels in .csv file
    final_filename = file_name + "_manual_label"
    output_path = os.path.join(outfile, f"{final_filename}.csv")
    output_path = os.path.normpath(output_path) # Nomalize path so forward slashes and backslashes aren't consistent
    
    X.to_csv(output_path)
    print(f"file saved at {output_path}")


if __name__ == "__main__":
    main()

# Plot amplitude according to fourier
# NOTE: Plotting is commented out, can be used to visualize the coloring for a given start and end short timeframe, about 5-6 minutes
# fig, ax = plt.subplots(figsize=(16, 5))  # Set a wide figure size
# ax.step(X.index, X[val], where='post')  # Step plot
# ax.scatter(X.index, X[val], c=X['color'], s=X['thickness'])  # Scatter plot overlay


# # show frequency value
# for i in range(len(X)):
#     freq_value = X[the_freq].iloc[i]
#     ax.text(X.index[i], X[val].iloc[i], f"{freq_value:.0f}", ha='center', va='bottom', fontsize=8, color='black')

# ax.set_xlabel('Datetime')
# ax.set_ylabel(val) 

# # Show ticks every 30 seconds
# ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# fig.autofmt_xdate()
# #plt.show()
# plt.savefig(outfile)
