import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Date formatting for graphs
import sys
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN, AgglomerativeClustering, Birch, HDBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure

# python explore_clustering.py kmeans
# python explore_clustering.py kmeans alpha
# python explore_clustering.py day

# 1. KMeans
GOOD_AMP = 0.5 # default: 271431 rows # 0.5: 1223 rows # 0.4: 2900 rows # 0.3: 7309 rows
MAX_FREQ = 30 # default: 271431 rows #25: 238741 # 30: 256052 rows # 35: 264147 rows
NUM_CLUSTERS = 2
ALPHA = 0.05
# 0.3/25: 6832 rows 


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

#infile = "../transformed_data_3fourier"
infile = sys.argv[1]
# "../transformed_data", "../transformed_data_2fourier", "../transformed_data_3fourier", "../transformed_data_5fourier", "../transformed_data_10fourier"

## NOTE MODIFY AS NEEDED
fig_name = "figures/rachel_nov21/ex2_fourier3_ampZ" # for day
X = pd.read_parquet(infile, engine='pyarrow')
print(X)

# Showing 5 minutes of April 30th only
X = X.loc[X.index.date == pd.to_datetime('2020-05-01').date()]

## EXAMPLE 1
# start_time = "2020-05-01 10:34:40.000"
# end_time = "2020-05-01 10:38:30.000"

# EXAMPLE 2
start_time = "2020-05-01 10:31:20.000"
end_time = "2020-05-01 10:32:20.000"

# filter for data points in this range
X = X.loc[start_time:end_time]

# NOTE MODIFY AS NEEDED plotted x and y dimensions, change to freqY/ampY, freqZ/ampZ, freq/amp as needed # VDBA
the_freq = 'freqZ' # X
the_amp = 'ampZ' # Y, also magnitude

## NOTE MODIFY AS NEEDED
val =  the_amp # value we're plotting by

    # 10-16: red
    # 22-25: green
categories = [(8 <= X[the_freq]) & (X[the_amp] >= 0.8),#& (X[the_freq] <= 18), # Green, Takeoff!
                ((0 <= X[the_freq]) & (X[the_freq] <= 10) & (X[the_amp] >= 0.2)) | ((X[the_freq] <= 15) & (X[the_amp] <= 0.5) & (X[the_amp] >= 0.2)), # Black, Slow Flight/End of Flight
                (10 < X[the_freq]) & (X[the_freq] < 15) & (X[the_amp] >= 0.5), # Red, Fast Flight/Start of Flight
                (18 <= X[the_freq]),# & (X[the_amp] >= 0.05)] # Feather Ruffling
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

print(X)

# plot amplitude according to fourier
fig, ax = plt.subplots(figsize=(16, 5))  # Set a wide figure size
ax.step(X.index, X[val], where='post')  # Step plot
ax.scatter(X.index, X[val], c=X['color'], s=X['thickness'])  # Scatter plot overlay


# show frequency value
for i in range(len(X)):
    freq_value = X[the_freq].iloc[i]
    ax.text(X.index[i], X[val].iloc[i], f"{freq_value:.0f}", ha='center', va='bottom', fontsize=8, color='black')

ax.set_xlabel('Datetime')
ax.set_ylabel(val)

# Show ticks every 30 seconds
ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
#plt.show()
plt.savefig(fig_name)


# if sys.argv[1] == 'kmeans':
#     X = process(X) # Chunkis the index
#     X = X.drop(columns=['ampY', 'freqY', 'ampZ', 'freqZ', 'amp', 'freq'])

#     NUM_ROWS = len(X)
#     #print(X)

#     # #quit()
#     model = make_pipeline(
#         MinMaxScaler(), # Standard Scalar, Function Transformer
#         KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
#     )

#     labels = model.fit_predict(X)

#     apply_pca = PCA(NUM_CLUSTERS)
#     plot_X = apply_pca.fit_transform(X) # reduce to two dimensions to plot
#     pca_df = pd.DataFrame(plot_X)
#     feature_loadings = pd.DataFrame(apply_pca.components_, columns=X.columns)

#     # plot the two PCA values
#     if len(sys.argv) > 2:
#         if sys.argv[2] == 'alpha': # DENSITY PLOT
#             # PCA
#             # plt.scatter(plot_X[:, 0], plot_X[:, 1], c=labels, cmap='rainbow', alpha=ALPHA)
#             # plt.title(f'{NUM_ROWS} rows,  KMeans={NUM_CLUSTERS}, PCA')

#             # plt.savefig(f'figures/rachel_oct24/amp{GOOD_AMP}freq{MAX_FREQ}_minmax_kmeans{NUM_CLUSTERS}_alpha{ALPHA}.png')

#             # No PCA
#             plt.scatter(X[the_freq], X[the_amp], c=labels, cmap='rainbow', alpha=ALPHA)
#             plt.title(f'{NUM_ROWS} rows,  KMeans={NUM_CLUSTERS}, PCA')
#             plt.xlabel(the_freq)
#             plt.ylabel(the_amp)
#             plt.show()
#             #plt.savefig(f'figures/rachel_oct24/no_PCA/no_X_plotY')
#             #plt.savefig(f'figures/rachel_oct24/no_PCA/added_nodim_X_amp{GOOD_AMP}freq{MAX_FREQ}_minmax_kmeans{NUM_CLUSTERS}_alpha{ALPHA}.png')
        
#     else: 
#         # PCA
#         # plt.scatter(plot_X[:, 0], plot_X[:, 1], c=labels, cmap='rainbow')#
#         # plt.title(f'{NUM_ROWS} rows,  KMeans={NUM_CLUSTERS}, PCA')
        
#         # plt.savefig(f'figures/rachel_oct24/amp{GOOD_AMP}freq{MAX_FREQ}_minmax_kmeans{NUM_CLUSTERS}.png')

#         # No PCA
#         plt.scatter(X[the_freq], X[the_amp], c=labels, cmap='rainbow')#
#         plt.title(f'{NUM_ROWS} rows,  KMeans={NUM_CLUSTERS}')
#         plt.xlabel(the_freq)
#         plt.ylabel(the_amp)
#         #plt.savefig(f'figures/rachel_oct24/no_PCA/no_Y_plotX')
#         plt.savefig(f'figures/rachel_oct24/no_PCA/added_nodim_X_amp{GOOD_AMP}freq{MAX_FREQ}_minmax_kmeans{NUM_CLUSTERS}.png')





# else:
#     print("command not found")


# too large for AffinityPropagation, SpectralClustering, Agglomerative Clustering
    # numpy._core._exceptions.MemoryError: Unable to allocate 274. GiB for an array with shape (36837258165,) and data type float64
# DBSCAN has long runtime

# StandardScaler + Birch crashed VSCode for me, but MinMax + Birch ran fine