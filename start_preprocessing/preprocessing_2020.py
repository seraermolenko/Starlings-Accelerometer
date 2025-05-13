import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# command: python 2020_preprocessing.py <filename> <output file>
# Note: pass absolute path, output saves as file named output so rename
# example python 2020_preprocessing.py C:\Users\rache\OneDrive\Documents\School_2024\Fall_2024\CMPT_416\data\EUST_001_BOX78_APR-30-2020.csv C:\Users\rache\OneDrive\Documents\School_2024\Fall_2024\CMPT_416\preprocessed-data
# EUST_002_BOX14-MAY-01-2020.csv 

# infile = sys.argv[1] # EUST_002_BOX14-MAY-01-2020
# folder = sys.argv[2]

# infile = "../data/EUST_003_BOX181_MAY-07-2020.csv"
# outfile = "../preprocessed-data/EUST_003_BOX181_MAY-07-2020"

# infile is path to .csv file, outfile is folder  to save to
def process_2020(data, filename, outfile):

    #print("File path: " + infile)
    #data = pl.scan_csv(infile)#[:10000] scan the whole file

    data = data.drop('MagX', 'MagY', 'MagZ', 'Batt. V. (V)', 'Metadata')


    data_datetime = data.select(pl.col('Date').str.strptime(pl.Datetime, "%d-%m-%Y"), pl.col('Time').str.strptime(pl.Time, "%H:%M:%S.%3f", strict=False)).collect()


    # Recombine datetime into dataframe
    data = data.collect()
    data_datetime = data_datetime.select(pl.col('Date').dt.combine(pl.col('Time')))
    # print(data_datetime.columns)
    data_datetime = data_datetime.rename({'Date': 'Datetime'})
    data_accel = data.select(pl.col(data.columns[3:]))
    tag_id = data.select(pl.col("Tag ID"))

    # recombine dataframes across columns
    final = pl.concat([tag_id, data_datetime, data_accel], how='horizontal')

    # rename Date to Datetime, convert from microseconds to milliseconds to match 2021
    #final['Datetime'] = final['Datetime'].astype('datetime64[ms]')
    #data_datetime = data_datetime.rename({'Date': 'Datetime'})

    #print(final) 
    #print(final.dtypes)
    file_path = os.path.join(outfile, filename.replace(".csv", ".parquet"))
    final.write_parquet(file_path) # returns Tag ID, Timestamp, X, Y, Z, Temp. (?C)
    #print("Saved results") # index to remove .csv from parquet file

# else:
#     print("Please use a 2020 file. Program terminating")