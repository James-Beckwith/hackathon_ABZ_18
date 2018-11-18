import hackathonSeismicAttributes as esa
import segyio
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skimfeat
import scipy.signal as scipysig
import SOM
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# define test file
file = 'C:\\Users\\py06j\\Documents\\work\\hackathon\\data\\FINAL_PSTM_FULL\\WG152D0002-00036A541-PSTM_FINAL_FILTERED-FULL_STK-249775484.sgy'

# read in segy
segy = segyio.open(file, iline=21, xline=233)

# n inlines
nCDP = len(segy._ilines)
#nT = len(segy._samples)
# limit to 6000 samples
nT=400
dt = segy._samples[1] - segy._samples[0]

# convert segy traces into a useable and displayable numpy array, limit to first 1000 samples
data = np.zeros([nT,nCDP])
a = 0
for trace in segy.trace:
    data[:,a] = np.asarray(trace[0:nT] * 1.0)
    a=a+1

# limit data laterally to speed up

# estimate seismic attributes
seisStructTensX, seisStructTensXY, seisStructTensY, cf, cfsig, envelope = esa.estimateSeismicAttributes(data, nCDP, nT, dt)

# convert attributes to vectors
envelopeV = np.reshape(envelope, [np.product(np.shape(envelope)), 1])
cfV = np.reshape(cf, [np.product(np.shape(cf)), 1])
cfsigV = np.reshape(cfsig, [np.product(np.shape(cfsig)), 1])
seisStructTensXV = np.reshape(seisStructTensX, [np.product(np.shape(seisStructTensX)), 1])
seisStructTensYV = np.reshape(seisStructTensY, [np.product(np.shape(seisStructTensY)), 1])
seisStructTensXYV = np.reshape(seisStructTensXY, [np.product(np.shape(seisStructTensXY)), 1])

# mask out bad data
#mask1 = (envelopeV > np.max(envelopeV)/100.0)
mask3 = (envelopeV < np.max(envelopeV) * 0.8)
mask2 = np.asarray([not np.isnan(cfV[i]) for i in range(len(cfV))])
maskFin = np.asarray([mask1[i][0] and mask2[i] and mask3[i][0] for i in range(len(mask1))])

# mask data and convert to approximately normal distributions
envelopeV2 = np.log(envelopeV[maskFin])
cfV2 = np.log(cfV[maskFin])
cfsigV2 = np.log(cfsigV[maskFin])
seisStructTensXV2 = np.log(seisStructTensXV[maskFin] + np.mean(seisStructTensXV[maskFin]) / 100.0)
seisStructTensYV2 = np.log(seisStructTensYV[maskFin] + np.mean(seisStructTensYV[maskFin]) / 100.0)
seisStructTensXYV2 = np.log(seisStructTensXYV[maskFin] + np.abs(np.min(seisStructTensXYV[maskFin])) + 1.0)

# feature scale and normalise
envelopeV3 = (envelopeV2 - np.mean(envelopeV2)) / np.std(envelopeV2)
cfV3 = (cfV2 - np.mean(cfV2)) / np.std(cfV2)
cfsigV3 = (cfsigV2 - np.mean(cfsigV2)) / np.std(cfsigV2)
seisStructTensXV3 = (seisStructTensXV2 - np.mean(seisStructTensXV2)) / np.std(seisStructTensXV2)
seisStructTensYV3 = (seisStructTensYV2 - np.mean(seisStructTensYV2)) / np.std(seisStructTensYV2)
seisStructTensXYV3 = (seisStructTensXYV2 - np.mean(seisStructTensXYV2)) / np.std(seisStructTensXYV2)

# train SOM on this input data
#concatenate features into 1 matrix
input_data = np.hstack([envelopeV3, cfV3, cfsigV3, seisStructTensXV3, seisStructTensYV3, seisStructTensXYV3])
print(np.shape(input_data))
som1 = SOM.SOM(input_data)
som1.run()

print('Finished SOM')

#TSNEout = TSNE(n_components=2, perplexity=5).fit_transform(input_data) - memory and time constraints - not used

kmeans = KMeans(n_clusters=6).fit(input_data)

# store labels in original array size - kmeans
clusts = np.zeros(np.shape(envelopeV))
clusts[maskFin] = np.expand_dims(kmeans.labels_, axis=1)
clusts_reshape = np.reshape(clusts,np.shape(envelope))

# store labels in original array size - kmeans
clusts_som = np.zeros(np.shape(envelopeV))
clusts_som[maskFin] = np.expand_dims(som1.finalNeuron, axis=1)
clusts_som_reshape = np.reshape(clusts,np.shape(envelope))


############
# plotting #
############
# dsplay data
fig, ax = plt.subplots()
cax = ax.imshow(data, vmin = np.min(data)/10, vmax = np.max(data)/10, aspect='auto', cmap=plt.cm.bone, extent=[segy._ilines[0], segy._ilines[-1], segy._samples[nT], segy._samples[0]])
ax.set_title('Seismic data')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(seisStructTensX, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]], vmin=1*10**5, vmax=1*10**7)
ax.set_title('Strucutre tensor X')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(seisStructTensXY, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]], vmin=-1*10**7, vmax=1*10**7)
ax.set_title('Structure tensor XY')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(seisStructTensY, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]], vmin=3*10**7, vmax=9*10**7)
ax.set_title('Structure tensor Y')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(cf, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[-1]], vmin=20, vmax=60)
ax.set_title('Centroid frequency')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)
cbar.ax.set_ylabel('Centroid frequency (Hz)')

fig, ax = plt.subplots()
cax = ax.imshow(cfsig, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[-1]], vmin=15, vmax=40)
ax.set_title('Centroid bandwidth')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)
cbar.ax.set_ylabel('Centroid bandwidth (Hz)')

fig, ax = plt.subplots()
cax = ax.imshow(np.abs(envelope), aspect='auto' , extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]], vmin=0, vmax=10000)
ax.set_title('Envelope of signal')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

plt.show()
