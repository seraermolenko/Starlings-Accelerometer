import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates # Date formatting for graphs


# Plot mag/freq over time as a day plot
# Input: Folder containing Preprocessed Parquet Files
# Output: .png of Plots of entire days of Time of day vs. Acceleration(X/Y/Z). One image per file per day is generated in the output folder
#   - one image is generated per day, with four graphs shown. Top-left is accX, Top-right is accY, bottom-left is accZ, bottom-right is empty.

# Example command:
# python DayPlot.py "../input_folder" "../output_folder"

def main():

    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])

    

    #errors if files provided as argv aren't valid
    if not os.path.isdir(sys.argv[1]):
        sys.exit(f"Input directory '{infile}' does not exist.")
    if not os.path.isdir(sys.argv[2]):
        print(f"{outfile} not found, creating folder")
        os.makedirs(outfile)

    # generate day plots for all files! NOTE this code will attempt to run on all files in directory
    # Ensure your input directory only have parquet files you'd like to generate plots for
    for file_path in Path(infile).iterdir():
        if file_path.is_file():
            try:

                data = pd.read_parquet(file_path, engine='pyarrow')
                file_name = Path(file_path).name # get filename
                print(f"Processing file: {file_name}")

                # For each file in input directory, generate dayplots
                generate_plot(data, file_name, outfile)

            except Exception as e:
                print(f"Skipping file {file_path.name}: {e}")
    

def generate_plot(data, file_name, outfile):
    unique_days = pd.unique(data.index.date)

    # Get each unique date from box data
    days_list = [day.strftime("%Y-%m-%d") for day in unique_days]

    # Generate accX, accY, accZ plots for each date
    for selected_day in days_list:

        #isolated_day = data[data['Datetime'].dt.date == pd.to_datetime(selected_day).date()]
        # Gather all datapoints for one day
        isolated_day = data[data.index.date == pd.to_datetime(selected_day).date()]
        
        # Create new plot for new day
        figure, axis = plt.subplots(2, 2, figsize=(30, 30)) # 4 to start, access by ax index
        # Create name of image!
        subtitle = file_name + "_" + selected_day + '_Data'
        figure.suptitle(subtitle)

        col_accel = ['accX', 'accY', 'accZ']
        for i, col in enumerate(col_accel): # for each column of interest, plot in subplot
            ax = axis[i // 2, i % 2] # get proper axis


            # Plot graph
            ax.plot(isolated_day.index, isolated_day[col])

            # format x-axis tick labels to show hourly
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1 hour intervals
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
            ax.tick_params(labelrotation=45) # rotate hour labels for clarity

            ax.set_xlabel('Time')
            ax.set_ylabel(col)
            #ax.set_title("April 30, 2020 " + col) # crowding other plots

        # end of for loop, display image, either have this commented in or plt.savefig() commented in to save as png
        #plt.show()

        # Save day plot into output folder
        output_path = os.path.join(outfile, f"{subtitle}.png")
        output_path = os.path.normpath(output_path)
        # Save the plot
        plt.savefig(output_path)
        print(f"image saved at {output_path}")




if __name__ == "__main__":
    main()


# # Example 1: Box 305, 1B May 1, 2021
# #outfile = "EUST_008_BOX305_APR-30-2021"

# # plt.plot(data['Timestamp'], data['accX'])
# # plt.xlabel("Timestamp")
# # plt.ylabel("accX")
# # plt.show()
# # Data shown from April 30th to May 3rd


# days = ["2020-04-30", "2020-05-01", "2020-05-02", "2020-05-03"]
# col_accel = ['accX', 'accY', 'accZ']

# # iterate over subplots
# for selected_day in days:

#     # Create new plot for new day
#     figure, axis = plt.subplots(2, 2, figsize=(30, 30)) # 4 to start, access by ax index
#     subtitle = 'EUST_008_BOX305 ' + selected_day + 'Data'
#     figure.suptitle(subtitle)

#     for i, col in enumerate(col_accel): # for each column of interest, plot in subplot
#         ax = axis[i // 2, i % 2] # get proper axis

#         # Take April 30th to start (about 5 million rows)
#         isolated_day = data[data['Timestamp'].dt.date == pd.to_datetime(selected_day).date()]
        
#         # Plot graph
#         ax.plot(isolated_day['Timestamp'], isolated_day[col])

#         # format x-axis tick labels to show hourly
#         ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1 hour intervals
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
#         ax.tick_params(labelrotation=45) # rotate hour labels for clarity

#         ax.set_xlabel("Timestamp")
#         ax.set_ylabel(col)
#         #ax.set_title("April 30, 2020 " + col) # crowding other plots

#     # end of for loop, displayList
#     #plt.show()

#     # Save day plot into figures folder
#     plt.savefig('figures/' + subtitle)
