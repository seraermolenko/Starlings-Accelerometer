
# Starling Accelerometer Data Analysis

Previous documentation for running files

## Usage

1. Clone the repository 
2. [Process data](#Processing Data with preprocessing.py)
3. [Visualize data Plotting.py](Visualizing with Plotting.py)
4. [Visualize data vizualize.py](Visualize data vizualize.py)

### Processing data with Preprocessing.py

### Visualizing with Plotting.py

This script assumes your data is already correctly formatted from Preprocessing.py.

**Run the script for plotting:**

1. In the for  loop at the bottom, comment out fft_df and apply_kmeans(fft_df)
	- COMMENT OUT HERE (FFT) (KNEANS)
	- make sure plotData(data) is ***uncommented**

2. Inside plotData(data), pick what dimension you would like to see plotted
	- COMMENT OUT HERE (PICK A DIMENSION)
	- If you plan on saving the data, adjust X, Y, Z, MEAN names in the plots

3. Inside plotData(data), pick what days you would like to ploy
	-  If you do not know the days of your file, use PICK A DAY

4. Run the script 
	 - python3 Plotting.py (FIle Path)
	 - Example: python3 Plotting.py ../Starling/EUST_001_BOX260_MAY-2.parquet
	 - Box 260 on May 2nd is Meghan's box. 

**Run the script for kmeans and fft:**

1. In the for  loop at the bottom, comment out plotData(data)
	- COMMENT OUT HERE (PLOTTING)
	- make sure fft_df and apply_kmeans(fft_df) are ***uncommented**

2.  Input your day, start time, and end time
	- use PICK A DAY if you need to see your file
	- Meghan's time: 
		- day = "2020-05-01"
		- start= "2020-05-01 10:34:40.000"
		- end= "2020-05-01 10:38:30.000"

3. Include or exclude fft plots 
	-  Recommended exclude if FFT is resampled by 1s or the interval is big and you ONLY want to see the Kmeans clustering 
	 - COMMENT OUT HERE (FFT PLOTS)

4. Run the script 
	 - python3 Plotting.py (FIle Path)
	 - Example: python3 Plotting.py ../Starling/EUST_001_BOX260_MAY-2.parquet
	 - Box 260 on May 2nd is Meghan's box. 


Notes: 
- The FFT results include both real and imaginary parts
- The FFT is done on each of the 100 data points in each second 
- The KMeans scatterplot is done on RAW un-condensed or filtered data point

Upcoming: 
- Filtering options, removing unrealistic frequencies 
- Understanding the maginitude vs amplitiude vs freqeuncy vs time plots 

## Previous documention 

Rachel: 2020_preprocessing (python 2020_preprocessing.py)
    - dropped 'MagX', 'MagY', 'MagZ', 'Batt. V. (V)', 'Metadata' columns
    - combined Date and Time into Datetime ('%d-%m-%Y %H:%M:%S%.3f')
    Infile: "../data/EUST_003_BOX181_MAY-07-2020.csv"
    Outfile: "EUST_003_BOX181_MAY-07-2020"

Sera: 2021 preprocessing 
    - dropped 'Battery Voltage (V)', 'Metadata' columns
    - converted Timestamp col type str -> Datetime ('%d-%m-%Y %H:%M:%S%.3f')
    Infile: "../data/EUST_002_BOX266_APR-29-2021.csv"
    Outfile: "EUST_002_BOX266_APR-29-2021"


### Visualize data vizualize.py

Sera: Preprocessing.py

Sera: Plotting.py


Rachel: visualize.py 
requirement: one preprocessed file (run 2020_preprocessing.py/2021_Starlings_Preprocessing.py depending on year)
Infile: "../preprocessed-data/EUST_008_BOX305_APR-30-2021"

    (python3 visualize.py generate_figures)
    - currently hardcoded, plots April 30, May 1, May 2, May 3 plots of accX, accY, accZ
    Output: plots of accX, accY, accZ for each day saved in 'figures/' folder

    (python3 visualize.py fourier)
    - output: graphs for each day split into accX, accY, accZ subplots and labelled per hour
    - creates 'acc' column, group data by CHUNK_LENGTH = 1 second as 'acclist' column then apply fourier transform, saving best frequency peak and magnitude (Hz) as feature in dataframe CHUNK_LENGTH
    - adds 8 columns amp and freq for accX, accY, accZ, acc (combination of other three features) to dataframe
    Outfile: "../transformed_data"

Rachel: explore_clustering.py
requirement: "../transformed_data" from visualize.py
Infile: "../transformed_data"

    (python3 explore_clustering.py kmeans)
    - removes extreme frequency cases, minimal magnitude cases, drops acceleration cols to cluster amp/freq only
    Output: scatter plot in "figures/rachel_oct24"
    (python3 explore_clustering.py kmeans alpha)
    Output: same as above but plots data density in "figures/rachel_oct24"


    (python3 explore_clustering.py birch)
    - currently trying MinMaxScaler, and PCA(2) for grahps 
    Output: plotted graphs in "figures/"


