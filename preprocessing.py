import polars as pl
import os 
import matplotlib.pyplot as plt
import pandas as pd
import sys
import polars as pl
from pathlib import Path

#from start_preprocessing import *
from start_preprocessing.preprocessing_2020 import process_2020
from start_preprocessing.Starlings_Preprocessing_2021 import process_2021

#NOTE: Description 
# Preprocessing 2021 and 2020 csv files and combining them into a parquet outfile 

#NOTE: make sure to specifcy input and output directory for original files and preprocessed files

#NOTE: Example run: python3 preprocessing.py ../Starling ../preprocessed-data

inputs = sys.argv[1:]

def renamer(c):
    map = {
        'X': 'accX',
        'Y': 'accY',
        'Z': 'accZ',
        'Temp. (?C)': 'Temp',
        'Date': 'Datetime',
        'Timestamp': 'Datetime',
        'Tag ID': 'TagID',
    }
    if c in map:
        return map[c]
    else:
        return c


def preprocessing(data):
    columns = data.collect_schema().names()

    for c in ['Battery Voltage (V)', 'Batt. V. (V)', 'Metadata', 'MagX', 'MagY', 'MagZ']:
        if c in columns:
            data = data.drop(c)

    # Double checking data format for 2020 files
    if 'Date' in columns and 'Time' in columns:
       data = data.with_columns(
           pl.col('Date').str.to_date('%d-%m-%Y'),
           pl.col('Time').str.to_time('%H:%M:%S%.3f'),
       )
       data = data.with_columns(pl.col('Date').dt.combine(pl.col('Time')))
       #data['Timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format= "%d-%m-%Y %H:%M:%S%.3f")

       # Dropping columns from the original daraframe 
       data = data.drop('Time') 
       data['Date'] = data['Date'].astype('datetime64[ms]') 

    if 'Timestamp' in columns:
       data = data.with_columns(
           pl.col('Timestamp').str.to_datetime('%d-%m-%Y %H:%M:%S%.3f'),
       )
    
    data = data.rename(renamer)

    return data



def main():
    print("Hello World!")
    if (len(sys.argv) != 3):
        sys.exit("Please pass two arguments for in-folder and out-folder")
    
    print(sys.argv[1])
    print(sys.argv[2])

    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])

    # errors if files provided as argv aren't valid
    if not os.path.isdir(sys.argv[1]):
        sys.exit(f"Input directory '{infile}' does not exist.")
    if not os.path.isdir(sys.argv[2]):
        sys.exit(f"'{outfile}' exists but is not a directory.")

    # process all csv files
    for csv_file in infile.glob("*.csv"):
        data = pl.scan_csv(csv_file)

        # try:
        # process 2020 files, save to outfile folder
        if 'MagX' in data.columns:
            process_2020(data, csv_file.name, outfile)
            print(f"Processed file {csv_file.name}")
        # process 2021 files, save to outfile folder
        elif 'Battery Voltage (V)' in data.columns:
            process_2021(data, csv_file.name, outfile)
            print(f"Processed file {csv_file.name}")

        # except:
        #     print(f"error running {csv_file}")
    print("done!")


if __name__ == "__main__":
    main()




# Iterating over the files in input directory
for file_path in inputs:
    files = []
    if file_path.endswith('.csv'):
        print(f'Opening file: {file_path}')

        #data = pd.read_csv(file_path)                               # Reading with Pandas
        data = pl.scan_csv(file_path)
        
        # De bugging
        #data.columns = data.columns.str.strip()                     # Removing leading or trailing spaces
        print(data)

        processed_file = preprocessing(data)
        processed_file.sink_parquet(file_path + '.parquet')
        #files.append(processed_file)

# Combining all of the files
#combined_processed = pd.concat(files).reset_index()         

# Writing to output file, (Parquet)
# Writing to a Parquet file
#combined_processed.to_parquet("outfile.parquet")

