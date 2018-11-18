import segyio
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skimfeat
import scipy.signal as scipysig

def estimateSeismicAttributes(data, nCDP, nT, dt, strucSigma=10, stftWinLen = 51, stftWinOver=50):

    # calculate structural tensor of stacked seismic data
    seisStructTensX, seisStructTensXY, seisStructTensY = skimfeat.structure_tensor(data, sigma=strucSigma)

    ''' calculate centroid frequency and centroid bandwidth
        1. Calculate spectrum of data
        2. Calculate frequency vector and range of frequencies to use
        3. form centroid frequency
        4. form standard deviation of centroid frequency
        '''

    # I can do better but this will have to do for now.
    cf = np.zeros([nT,nCDP])
    cfsig = np.zeros([nT, nCDP])
    for i in range(nCDP):
        freqs, stftTimes, dataTimeFreq = scipysig.stft(data[:,i], fs = 1000.0/dt, nperseg = stftWinLen, noverlap=stftWinOver)
        # calculate centroid frequency and centroid bandwidth adn store in array
        for j in range(len(stftTimes)):
            sumSpecs = np.sum(np.abs(dataTimeFreq[:,j]))
            cf[j,i] = np.sum(freqs * np.abs(dataTimeFreq[:,j])) / sumSpecs
            cfsig[j,i] = (np.sum((freqs - cf[j,i]) ** 2 * np.abs(dataTimeFreq[:,j])) / sumSpecs) ** 0.5
        # output progress
        if np.mod(i,100)==0:
            print('%i traces of %i total transformed' % (i, nCDP))

    # form envelope of signal
    envelope = scipysig.hilbert(data, axis=0)

    return seisStructTensX, seisStructTensXY, seisStructTensY, cf, cfsig, np.abs(envelope)
