import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Date formatting for graphs
import pandas as pd
from scipy.fft import fft, ifft, fftfreq, rfftfreq, rfft
from scipy.signal import find_peaks
import numpy as np
import sys
import os
from pathlib import Path

# python visualize.py "../trythis" "../postvisualize"

# Note: takes about 30s to run
CHUNK_LENGTH = 3
SAMPLE_RATE = 100 # 100 samples/second
N = SAMPLE_RATE * CHUNK_LENGTH # 100 for 1 second


# infile = "../preprocessed-data/EUST_008_BOX305_APR-30-2021"
# outfile = "../transformed_data_15fourier" 

# Apply fourier transformation to each row.
# Input: row, function called using .apply
# Output: column values for max amplitude and frequency for chosen column from (accX, accY, accZ, acc)
def apply_fourier(row):

    fourier = rfft(row)
    frequency = rfftfreq(N, 1 / SAMPLE_RATE)

    # Calculate N/2 to normalize the FFT output
    normalized_amplitude = np.abs(fourier)/(N/2)

    # find strongest peaks
    peaks, _ = find_peaks(normalized_amplitude, height=0) # peaks returns indices of max peaks

    # print(normalized_amplitude[peaks]) # returns list of amplitudes
    # print(np.argmax(normalized_amplitude[peaks])) # get maximum amplitude
    best_index = np.argmax(normalized_amplitude[peaks])
    best_amplitude = normalized_amplitude[peaks][best_index]
    best_frequency = frequency[peaks][best_index]

    return best_amplitude, best_frequency


# Takes one preprocessed file as input, plots each day separately into accX, accY, accZ subplots and saves into figures

# Example 1 in EUST accelerometer signature analysis powerpoint that Tony sent
# May 1, 2020 10:34:40 to 10:38:30
# start_time = "2020-05-01 10:34:40.000"
# end_time = "2020-05-01 10:38:30.000"
# filter for data points in this range
# XXX isolated_time = data[(data['Timestamp'] >= start_time) & (data['Timestamp'] <= end_time)]
def run_fourier(data, infile, outfile):
    isolated_time = data # process entire file
    # FOURIER TRANSFORM
    isolated_time = isolated_time.copy()
    isolated_time.loc[:, 'acc'] = np.sqrt(isolated_time['accX']**2 + isolated_time['accY']**2 + isolated_time['accZ']**2)

    # group into chunks by 1s to start, timestamp is in [ms]!
    #list_interest = 'acclist'

    # Use df.floor instead of dt.round to ensure all arrays are length 100 per CHUNK_LENGTH = 1 second
    isolated_time['Chunk'] = isolated_time['Datetime'].dt.floor(f'{CHUNK_LENGTH}s')
    grouped_df = isolated_time.groupby('Chunk').agg({'accX': 'mean', 'accY': 'mean', 'accZ': 'mean', 'acc': 'mean'})

    # calculate fourier best amplitude and frequency for each of columns
    for index, col in enumerate(['accX', 'accY', 'accZ', 'acc']):
        label = ['X', 'Y', 'Z', '']
        grouped_x = isolated_time.groupby('Chunk')[col].apply(lambda data: np.array(data, dtype=np.float64)) # each array has 99 elements according to len()
        grouped_df['acclist'] = grouped_x
        
        # drop first and last row as they aren't full seconds and don't have 100 measurements
        grouped_df = grouped_df.iloc[:-1]# drop last row which has one element
        results = grouped_df['acclist'].apply(apply_fourier) # returns (max amplitude, max frequency)

        # add amplitude and frequency to their respective columns
        grouped_df['amp' + label[index]] = results.apply(lambda x: x[0])
        grouped_df['freq' + label[index]] = results.apply(lambda x: x[1])

    #print(results)
    print(grouped_df)
    #file_path = os.path.join(outfile, filename) # write to new directory
    grouped_df.to_parquet(outfile)
    print(f"Saved files to {outfile}")

# one file at a time
# python visualize <path to infile> <path to outfile>
def main():
    print("Hello World!")
    if (len(sys.argv) != 3):
        sys.exit("Please pass two arguments for in-folder and out-folder")
    
    
    # print(sys.argv[1])
    # print(sys.argv[2])
    infile = sys.argv[1]
    outfile = sys.argv[2]


    data = pd.read_parquet(infile, engine='pyarrow')
    # print(data.head)
    # print(data.columns.values.tolist())
    run_fourier(data, infile, outfile)

    # infile = Path(sys.argv[1])
    # outfile = Path(sys.argv[2])

    # # errors if files provided as argv aren't valid
    # if not os.path.isdir(sys.argv[1]):
    #     sys.exit(f"Input directory '{infile}' does not exist.")
    # if not os.path.isdir(sys.argv[2]):
    #     sys.exit(f"Output directory '{outfile}' does not exist.")

    # # process all parquet files
    # for parquet_file in infile.glob("*.parquet"):
    #     data = pd.read_parquet(infile, engine='pyarrow')

    #     # applies fourier transform and saves file into provided output directory
    #     run_fourier(data, infile, outfile)
    #     print(f"Applied fourier to file {parquet_file.name}")


    print("done!")


if __name__ == "__main__":
    main()




# if sys.argv[1] == "generate_figures":
#     # Example 1: Box 305, 1B May 1, 2021
#     #outfile = "EUST_008_BOX305_APR-30-2021"

#     # plt.plot(data['Timestamp'], data['accX'])
#     # plt.xlabel("Timestamp")
#     # plt.ylabel("accX")
#     # plt.show()
#     # Data shown from April 30th to May 3rd


#     days = ["2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03"]
#     col_accel = ['accX', 'accY', 'accZ']

#     # iterate over subplots
#     for selected_day in days:

#         # Create new plot for new day
#         figure, axis = plt.subplots(2, 2, figsize=(30, 30)) # 4 to start, access by ax index
#         subtitle = 'EUST_008_BOX305 ' + selected_day + 'Data'
#         figure.suptitle(subtitle)

#         for i, col in enumerate(col_accel): # for each column of interest, plot in subplot
#             ax = axis[i // 2, i % 2] # get proper axis

#             # Take April 30th to start (about 5 million rows)
#             isolated_day = data[data['Timestamp'].dt.date == pd.to_datetime(selected_day).date()]
            
#             # Plot graph
#             ax.plot(isolated_day['Timestamp'], isolated_day[col])

#             # format x-axis tick labels to show hourly
#             ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1 hour intervals
#             ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
#             ax.tick_params(labelrotation=45) # rotate hour labels for clarity

#             ax.set_xlabel("Timestamp")
#             ax.set_ylabel(col)
#             #ax.set_title("April 30, 2020 " + col) # crowding other plots

#         # end of for loop, displayList
#         #plt.show()

#         # Save day plot into figures folder
#         plt.savefig('figures/' + subtitle)
