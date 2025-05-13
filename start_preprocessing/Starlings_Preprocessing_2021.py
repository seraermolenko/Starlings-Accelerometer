import polars as pl
import os
# infile = "../data/EUST_008_BOX305_APR-30.csv"
# outfile = "../preprocessed-data/EUST_008_BOX305_APR-30-2021"

# infile is path to .csv file, outfile is folder  to save to
def process_2021(data, filename, outfile):

    #data = pl.scan_csv(infile)#[:10000]

    # Removing battery voltage 
    data = data.drop('Battery Voltage (V)', 'Metadata')
    # Double checking data format 
    #print(data.select(pl.col('Timestamp')).collect())

    # Convert Timestamp str to Datetime data type
    data = data.with_columns(pl.col('Timestamp').str.to_datetime('%d-%m-%Y %H:%M:%S%.3f'))
    data = data.rename({'Timestamp': 'Datetime'})


    # Looking for any meta data 
    #print(data.select(pl.col('Metadata')).collect())
    #metaData = data.filter(pl.col('Metadata').is_not_null())
    #print(metaData.select(pl.col('Metadata')).collect())
    #print(data.collect())
    # collect data before writing
    data = data.collect()
    print(data)
    file_path = os.path.join(outfile, filename.replace(".csv", ".parquet"))
    data.write_parquet(file_path) # returns Tag ID, Timestamp, X, Y, Z, Temp. (?C)





