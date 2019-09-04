import segyio
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skimfeat
import scipy.signal as scipysig


''' Estimate some pre-defined seismic attributes from an input 2D numpy array. Attribures estimated are:
    structural tensor (X, Y, XY)
    Envelope of signal
    Centroid frequency
    centroid bandwidth

    Required inputs: data - 2D numpy array of shape nT x nCDP
                     dt - spacing between two consecutive time samples (assumes regular sampling)
    Optional inptus - strucSigma - hyper-parameter for strucutral tensor estimation
                      stftWinLen - length of the rolling window used to Fourier transform the data (in samples)
                      stftWinOver - overlap of rolling windows for Fourier transform (in samples)
'''
def estimateSeismicAttributes(data, dt, strucSigma=10, stftWinLen = 51, stftWinOver=50):

    # calculate structural tensor of stacked seismic data
    seisStructTensX, seisStructTensXY, seisStructTensY = skimfeat.structure_tensor(data, sigma=strucSigma)

    # I can do better but this will have to do for now.
    cf = np.zeros([nT,nCDP])
    cfsig = np.zeros([nT, nCDP])
    for i in range(nCDP):
        freqs, stftTimes, dataTimeFreq = scipysig.stft(data[:,i], fs = 1000.0/dt, nperseg = stftWinLen, noverlap=stftWinOver)
        # calculate centroid frequency and centroid bandwidth and store in array
        for j in range(len(stftTimes)):
            sumSpecs = np.sum(np.abs(dataTimeFreq[:,j]))
            cf[j,i] = np.sum(freqs * np.abs(dataTimeFreq[:,j])) / sumSpecs
            cfsig[j,i] = (np.sum((freqs - cf[j,i]) ** 2 * np.abs(dataTimeFreq[:,j])) / sumSpecs) ** 0.5
        # output progress
        if np.mod(i,100)==0:
            print('%i traces of %i total transformed' % (i, nCDP))

    # form envelope of signal
    envelope = np.abs(scipysig.hilbert(data, axis=0))

    return seisStructTensX, seisStructTensXY, seisStructTensY, cf, cfsig, envelope
