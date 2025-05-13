# starlings-accelerometer

This project focuses on preporcessing and analyzing accelerometer data collected from Starlings to identify distinct behaviors such as flying, resting, or foraging. This project was executed by Serafima Ermolenko and Rachel Lagasse, working in collaboration with Professor Greg Baker, and Professor Tony Williams' lab at Simon Fraser University. 

Our goal was to transform approximately 20GB of raw CSV files into meaningful insights, supporting biological research on starling activity patterns. This project includes data preprocessing, behavior labeling, and analysis. 


This program first intakes raw accelromter data, and cleanes it into a unified parquet files which contian data collected at 100 samples per second. We then use FFT anaylsis to obtain frequency and magnitude data. This process outputs two csv files, containing maginitude and frequency values for X axsis and VeDBA repsectivley. From here, our program allows for different forms of visualzation, categorized into scripts that give you labelled or unlabblled output. These visulazation forms inlcude plotting magnitude and frequency, kmeans clustering, decision tree, and an adjustable manual decision tree. 

**I worked on this project with 1 other student Rachel Lagasse and under supervision from professeors Greg Baker and Tony Williams.**


## Prerequisites

- Python 3.x 

Run
- `pip install -r requirements.txt`

This will pip install these required Python libraries:
- `pandas`
- `numpy`
- `matplotlib` 
- `seaborn` 
- `scikit-learn`

## Usage

1. Clone the repository 

2. [Process data](#processing-data)
- preprocessing.py
- fft.py

3. [Explore data -> Unlablled Output](#explore-data--unlabeled-output)
- kmeans.py
- plotting.py 

4. [Explore data -> Labelled Output](#explore-data--labelled-output)
- decision_tree.py
- manual_decision_tree.py

![Diagram](diagra/diagram.png)
### Processing Data

#### preprocessing.py

NOTE: This script assumes the input csv files are in either 2020 or 2021 format that were given to us, and may need adjustment if future csv files include changes in collumn names or new collumns. 

Description: 
This script processes CSV files using Polars, applies preprocessing for data, and saves the results as Parquet files. 

1. Preprocessing Functions:
- renamer: Renames columns from the raw CSV to standardized names.
- preprocessing: Cleans the dataset (e.g., drops unwanted columns, handles date/time).
- process_2020: Handles 2020-specific files, cleaning and adjusting timestamp formats.
- process_2021: Handles 2021-specific files, similarly cleaning and adjusting timestamps.

2. Main Execution:
- Checks for valid input and output directories.
- Iterates through all CSV files in the input directory, processes each file, and saves the results to the output directory in Parquet format.

How to run:
1. Process All Files in a Folder 
```python3 preprocessing.py <path_to_your_infile_csv_folder> <path_to_your_outfile_folder>```
2. Process a Specific File 
```python3 preprocessing.py <path_to_your_infile_csv_file> <path_to_your_outfile_folder>```

Outputs: 
For each processed file:
- A Parquet file is saved in the output folder with the same name as the original CSV file, but with a .parquet extension.

These Parquet files contain the following data:
- Tag ID: Unique identifier for the tag.
- Datetime: The combined timestamp of the date and time, converted to milliseconds.
- accX: Acceleration along the X-axis.
- accY: Acceleration along the Y-axis.
- accZ: Acceleration along the Z-axis.
- Temp: Temperature in Celsius.

Notes: 
For 2021 Files:
- Columns such as Battery Voltage (V) and Metadata are removed from the data.
- The Timestamp is converted to a datetime format.
- The preprocessed data is written to a Parquet file for further analysis.
For 2020 Files:
- The script removes MagX, MagY, MagZ, Batt. V. (V), and Metadata columns.
- The Date and Time columns are combined into a Datetime column, formatted correctly.

#### fft.py

NOTE: 
This script assumes data was preprocessed. If it was not, run preprocessing.py. 

Description: 
This script processes post-preprocessing accelerometer data stored in .parquet format. The data is analyzed using Fast Fourier Transform (FFT) to extract frequency-domain features. The script focuses on two key outputs:
- VeDBA: Vectorial Dynamic Body Acceleration (a measure of overall movement intensity).
- accZ: Z-axis acceleration data.

For each second (100 data points), the script performs FFT, and then averages the results over two-second intervals. Users can adjust the averaging interval or limit the analysis to specific daytime hours by modifying the start and end times in the script.

Features: 
- FFT Analysis: Converts time-domain accelerometer signals into frequency-domain data for analysis.
- Daytime Filtering: Allows analysis to be limited to specific time periods (e.g., 6 AM to 9 PM).
- Averaging: Groups FFT results over adjustable intervals (default: 2 seconds).
- Customizable Thresholds: Filters frequencies and magnitudes to focus on meaningful data.

Outputs
Two CSV files are generated for each day:
- fft_VeDBA_results: Contains FFT analysis results for VeDBA.
- fft_accZ_results: Contains FFT analysis results for Z-axis acceleration.
Files are saved in separate directories: fft_VeDBA_results and fft_accZ_results.

How to run:
1. Process All Files in a Folder 
```python3 fft.py <path_to_your_parquet_folder>```
2. Process a Specific File
```python3 fft.py <path_to_your_parquet_file>```

Adjustments: 
Time Period: Modify the start and end variables in the script to analyze specific time intervals.
```
start = f"{day} 06:00:00.000"  # Example: Start at 6 AM
end = f"{day} 21:00:00.000"    # Example: End at 9 PM
```
Averaging Interval: Change the chunk_length variable to adjust the averaging duration:
```
chunk_length = 2  # Average over 2 seconds (default)
```

Notes
The script filters FFT results to include:
- Frequencies ≤ 30 Hz
- Magnitudes > 0.5
Wworking directory must contain subdirectories for outputs named:
- fft_VeDBA_results
- fft_accZ_results

Output Files
For each day processed:
- fft_VeDBA_results/{day}_fft_VeDBA.csv:
    - Columns: datetime, Frequency, Magnitude
- fft_accZ_results/{day}_fft_accZ.csv:
    - Columns: datetime, Frequency, Magnitude


Output Collumns in CSV: 
- DateTime – The timestamp for the recorded data.
- Magnitude – The magnitude value from the FFT analysis.
- Frequency – The frequency value from the FFT analysis.

### Explore Data -> Unlablled Output

####  kmeans.py

#### plotting.py 

### Explore data -> Labelled Output

#### decision_tree.py

NOTE: 
This script assumes data was preprocessed. If it was not, run preprocessing.py. 

Description: 
This script processes FFT data to classify the behavior of starlings based on frequency and magnitude. It applies a decision tree approach to categorize behaviors such as "Idle," "Flying," "Ruffle," "Feeding," "Foraging," and "Not Flying." The data is input from a CSV file containing FFT results.

How to run:
```python3 decision_tree.py <path_to_your_csv_file>```

Output: 
- A plot of magnitude against time, saved as (Datetime)_decision_tree.png in the decision_tree_results folder.
- A scatter plot showing the relationship between frequency and magnitude, color-coded by behavior, is saved as scatter_plot.png in the decision_tree_results folder.

Behavior Classification: 
The bird behaviors are classified based on frequency and magnitude thresholds. A DataFrame with the classified behaviors is printed to the console. The possible behaviors include:
- Idle: If the magnitude is below 0.5.
- Flying: If the frequency is between 10-20 Hz, and the bird has been ruffling for more than 2 seconds.
- Ruffle: If the frequency is between 10-20 Hz but the bird hasn't been flying yet.
- Feeding: If the frequency is between 0.5-4 Hz and the bird has been feeding for less than 8 seconds.
- Foraging: If the bird has been feeding for more than 8 seconds.
- Not Flying: If the frequency doesn't match any other conditions.

#### manual_decision_tree.py

