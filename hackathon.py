import hackathonSeismicAttributes as esa
import segyio
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skimfeat
import scipy.signal as scipysig

# define test file
file = 'C:\\Users\\py06j\\Documents\\work\\hackathon\\data\\FINAL_PSTM_FULL\\WG152D0002-00036A541-PSTM_FINAL_FILTERED-FULL_STK-249775484.sgy'

# read in segy
segy = segyio.open(file, iline=21, xline=233)

# n inlines
nCDP = len(segy._ilines)
#nT = len(segy._samples)
# limit to 1000 samples
nT=1000
dt = segy._samples[1] - segy._samples[0]

# convert segy traces into a useable and displayable numpy array, limit to first 1000 samples
data = np.zeros([nT,nCDP])
a = 0
for trace in segy.trace:
    data[:,a] = np.asarray(trace[0:1000] * 1.0)
    a=a+1

# estimate seismic attributes
seisStructTensX, seisStructTensXY, seisStructTensY, cf, cfsig, envelope = esa.estimateSeismicAttributes(data, nCDP, nT, dt)

# convert attributes to vectors
envelopeV = np.reshape(envelope, [np.product(np.reshape(envelope)), 1])
cfV = np.reshape(cf, [np.product(np.reshape(cf)), 1])
cfsigV = np.reshape(cfsig, [np.product(np.reshape(cfsig)), 1])
seisStructTensXV = np.reshape(seisStructTensX, [np.product(np.reshape(seisStructTensX)), 1])
seisStructTensYV = np.reshape(seisStructTensYV, [np.product(np.reshape(seisStructTensYV)), 1])
seisStructTensXYV = np.reshape(seisStructTensXYV, [np.product(np.reshape(seisStructTensXYV)), 1])

# mask out bad data
mask1 = envelopeV > np.max(envelopeV)/1000.0
mask2 = [not np.isnan(cfV[i]) in range(len(cfV))]
maskFin = mask1 and mask2

# mask data and convert to approximately normal distributions
envelopeV2 = np.log(envelopeV[maskFin])
cfV2 = np.log(cfV[maskFin])
cfsigV2 = np.log(cfsigV[maskFin])
seisStructTensXV2 = np.log(seisStructTensXV[maskFin] + np.mean(seisStructTensXV[maskFin]) / 100.0)
seisStructTensYV2 = np.log(seisStructTensYV[maskFin] + np.mean(seisStructTensYV[maskFin]) / 100.0)
seisStructTensXYV2 = np.log(seisStructTensXYV[maskFin] + np.mean(seisStructTensXYV[maskFin]) / 100.0)

# feature scale and normalise
envelopeV3 = (envelopeV2 - np.mean(envelopeV2)) / np.std(envelopeV2)
cfV3 = (cfV2 - np.mean(cfV2)) / np.std(cfV2)
cfsigV3 = (cfsigV2 - np.mean(cfsigV2)) / np.std(cfsigV2)
seisStructTensXV3 = (seisStructTensXV3 - np.mean(seisStructTensXV2)) / np.std(seisStructTensXV2)
seisStructTensYV3 = (seisStructTensYV3 - np.mean(seisStructTensYV2)) / np.std(seisStructTensYV2)
seisStructTensXYV3 = (seisStructTensXYV3 - np.mean(seisStructTensXYV2)) / np.std(seisStructTensXYV2)

# train SOM on this input data


############
# plotting #
############
# dsplay data
fig, ax = plt.subplots()
cax = ax.imshow(data, vmin = np.min(data)/10, vmax = np.max(data)/10, aspect='auto', cmap=plt.cm.bone, extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]])
ax.set_title('Seismic data')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(seisStructTensX,vmin = np.min(seisStructTensX)/10, vmax = np.max(seisStructTensX)/10, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]])
ax.set_title('Strucutre tensor X')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(seisStructTensXY,vmin = np.min(seisStructTensXY)/10, vmax = np.max(seisStructTensXY)/10, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]])
ax.set_title('Structure tensor XY')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(seisStructTensY, vmin = np.min(seisStructTensY)/10, vmax = np.max(seisStructTensY)/10, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]])
ax.set_title('Structure tensor Y')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

fig, ax = plt.subplots()
cax = ax.imshow(cf, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], stftTimes[0], stftTimes[-1]])
ax.set_title('Centroid frequency')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)
cbar.ax.set_ylabel('Centroid frequency (Hz)')

fig, ax = plt.subplots()
cax = ax.imshow(cfsig, aspect='auto', extent=[segy._ilines[0], segy._ilines[-1], stftTimes[0], stftTimes[-1]])
ax.set_title('Centroid bandwidth')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)
cbar.ax.set_ylabel('Centroid bandwidth (Hz)')

fig, ax = plt.subplots()
cax = ax.imshow(np.abs(envelope), aspect='auto' , extent=[segy._ilines[0], segy._ilines[-1], segy._samples[0], segy._samples[nT]])
ax.set_title('Envelope of signal')
ax.set_xlabel('CDP')
ax.set_ylabel('TWT (ms)')
cbar = fig.colorbar(cax)

plt.show()
