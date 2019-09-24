import attr_extraction as esa
import segyio
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skimfeat
import scipy.signal as scipysig
#import SOM
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer

####################################
########## DATA I/O ################
####################################
# define test file
file = '/home/bp/test_data/FINAL_PSTM_FULL/WG152D0002-00067A559-PSTM_FINAL_FILTERED-FULL_STK-249775550.sgy'

# read in segy
segy = segyio.open(file, iline=21, xline=233)

# n inlines
nCDP = len(segy._ilines)
#nT = len(segy._samples)
# limit to nT samples
nT=400
dt = segy._samples[1] - segy._samples[0]

# convert segy traces into a useable and displayable numpy array, limit to first 1000 samples
data = np.zeros([nT, nCDP])
a = 0
for trace in segy.trace:
    data[:, a] = np.asarray(trace[0:nT] * 1.0)
    a = a+1

##### FEATURE EXTRACTION ########
# estimate seismic attributes
outFeatures = esa.estimateSeismicAttributes(data, dt, stftWinLen=35, stftWinOver=34)
outFeatureKeys = list(outFeatures.keys())
# deal with non-pixel based features values
if 'spectrogram' in outFeatureKeys:
    outFeatureKeys.remove('spectrogram')
    outFeatureKeys.remove('freqs')
    outFeatureKeys.remove('stftTimes')
    spectrogram = outFeatures.pop('spectrogram')
    freqs = outFeatures.pop('freqs')
    stftTimes = outFeatures.pop('stftTimes')

# store original shape of features
originalShape = outFeatures[outFeatureKeys[0]].shape

# # convert attributes to vectors
for key in outFeatureKeys:
    outFeatures[key] = np.reshape(outFeatures[key], [np.product(np.shape(outFeatures[key])), 1])

####### FEATURE SCALING ########
# mask data and convert to approximately normal distributions - some are still far from normally distributed
for key in outFeatureKeys:
    outFeatures[key] = PowerTransformer(method='yeo-johnson').fit_transform(outFeatures[key])

######### CLUSTERING ################
# #concatenate features into 1 matrix. This could/should be done much more efficiently
input_data = np.hstack(list(outFeatures.values()))

# # train SOM on this input data
# print(np.shape(input_data))
# som1 = SOM.SOM(input_data)
# som1.run()
# print('Finished SOM')

# # Tran TSNE
#TSNEout = TSNE(n_components=2, perplexity=5).fit_transform(input_data) - memory and time constraints - not used

# run kmeans clustering
kmeans = KMeans(n_clusters=6).fit(input_data)

''' function to store clusters into an array of a given size. It is assumed that the array will be expanded across the first axis'''
def store_labels_original_size(labels, original_size, label_size):
    labels_vec = np.zeros(label_size)
    labels_vec = np.expand_dims(labels, axis=1)
    original_size_labels = np.reshape(labels_vec, original_size)
    return original_size_labels

# store labels in original array size - kmeans
clusts_reshape = store_labels_original_size(kmeans.labels_, originalShape, np.shape(outFeatures[outFeatureKeys[0]]))

# # store labels in original array size - SOM
# clusts_som_reshape = store_labels_original_size(som1.finalNeuron, originalShape, np.shape(outFeatures[outFeatureKeys[0]]))
