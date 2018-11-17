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
nT = len(segy._samples)
dt = segy._samples[1] = segy._samples[0]

# convert segy traces into a useable and displayable numpy array
data = np.zeros([nT,nCDP])
a = 0
for trace in segy.trace:
    data[:,a] = np.asarray(trace * 1.0)
    a=a+1

# dsplay data
plt.imshow(data,vmin = np.min(data)/10, vmax = np.max(data)/10, aspect='auto', cmap=plt.cm.bone)

# calculate structural tensor of stacked seismic data
seisStructTensX, seisStructTensXY, seisStructTensY = skimfeat.structure_tensor(data,sigma=1)

plt.figure()
plt.imshow(seisStructTensX,vmin = np.min(seisStructTensX)/10, vmax = np.max(seisStructTensX)/10, aspect='auto')

plt.figure()
plt.imshow(seisStructTensXY,vmin = np.min(seisStructTensXY)/10, vmax = np.max(seisStructTensXY)/10, aspect='auto')

plt.figure()
plt.imshow(seisStructTensY,vmin = np.min(seisStructTensY)/10, vmax = np.max(seisStructTensY)/10, aspect='auto')

''' calculate centroid frequency and centroid bandwidth
    1. Calculate spectrum of data
    2. Calculate frequency vector and range of frequencies to use
    3. form centroid frequency
    4. form standard deviation of centroid frequency
    '''

# # I can do better but this will have to do for now.
# cf = np.zeros([nT,nCDP])
# for i in range(nCDP):
#     freqs, stftTimes, dataTimeFreq = scipysig.stft(data[:,i], fs = 1000.0/dt, nperseg = 50, noverlap=49)
#     # calculate centroid frequency and centroid bandwidth adn store in array
#     for j in range(len(stftTimes)):
#         sumSpec = np.sum(np.abs(dataTimeFreq[:,j]))
#         cf[j,i] = np.sum(freqs * np.abs(dataTimeFreq[:,j])) / sumSpec
#         cfsig[j,i] = np.sum((freqs - cf[j,i]) ** 2 * np.abs(dataTimeFreq[:,j])) / sumSpecs

# form envelope of signal
envelope = scipysig.hilbert(data, axis=0)

plt.figure()
plt.imshow(np.abs(envelope), aspect='auto')

plt.show()
