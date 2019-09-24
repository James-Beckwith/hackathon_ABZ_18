import numpy as np
import skimage.feature as skimfeat
import scipy.signal as scipysig

''' Function to calculate centroid frequency and frequency bandwidth given an input spectrogram of shape nF, nT'''
def calculateCF(timeVec, freqVec, data):
    nT = data.shape[1]
    cf = np.zeros([nT])
    cfsig = np.zeros([nT])
    for j in range(len(timeVec)):
        sumSpecs = np.sum(data[:, j])
        if np.abs(sumSpecs) < 1e-6:
            cf[j] = 0
            cfsig[j] = 0
        else:
            cf[j] = np.sum(freqVec * data[:, j]) / sumSpecs
            cfsig[j] = (np.sum((freqVec - cf[j]) ** 2 * data[:, j]) / sumSpecs) ** 0.5
    return cf, cfsig

''' return the next power of 2 greater than the input'''
def nextpow2(input):
    return int(2 ** np.ceil(np.log2(input)))

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
def estimateSeismicAttributes(data, dt, strucSigma=10, stftWinLen=51, stftWinOver=50, features=('structural tensor', 'envelope', 'cf', 'spectrogram')):

    # initialise output dictionary of features
    outFeatures = {}

    nT, nCDP = data.shape

    # calculate structural tensor of stacked seismic data
    if 'structural tensor' in features:
        outFeatures['structTensX'], outFeatures['structTensXY'], outFeatures['structTensY'] = skimfeat.structure_tensor(data, sigma=strucSigma)

    # form envelope of signal
    if 'envelope' in features:
        outFeatures['envelope'] = np.abs(scipysig.hilbert(data, axis=0))

    # create flag for features that need to be calculated by looping over traces and preallocate arrays
    traceFeatures = False
    if 'cf' in features:
        NFFT = nextpow2(stftWinLen)
        outFeatures['cf'] = np.zeros([nT, nCDP])
        outFeatures['cfsig'] = np.zeros([nT, nCDP])
        traceFeatures = True
    if 'spectrogram' in features:
        NFFT = nextpow2(stftWinLen)
        outFeatures['spectrogram'] = np.zeros([NFFT // 2 + 1, nT, nCDP], dtype='complex128')
        traceFeatures = True

    # calculate trace features if necessary
    if traceFeatures:
        print('computing trace specific features')
        for i in range(nCDP):
            # calculate spectrogram
            if 'spectrogram' in features:
                freqs, stftTimes, outFeatures['spectrogram'][:, :, i] = scipysig.stft(data[:, i], fs=1000.0/dt, nperseg=stftWinLen, noverlap=stftWinOver, nfft=NFFT)
               # calculate centroid frequency and centroid bandwidth and store in array
            if 'cf' in features or 'cfsig' in features:
                if 'spectrogram' not in features:
                    freqs, stftTimes, dataTimeFreq = scipysig.stft(data[:, i], fs=1000.0 / dt, nperseg=stftWinLen,
                                                                   noverlap=stftWinOver)
                    outFeatures['cf'][:, i], outFeatures['cfsig'][:, i] = calculateCF(stftTimes, freqs, dataTimeFreq)
                else:
                    outFeatures['cf'][:, i], outFeatures['cfsig'][:, i] = calculateCF(stftTimes, freqs, np.abs(outFeatures['spectrogram'][:, :, i]))
            # output progress
            if np.mod(i, 100) == 0:
                print('%i traces of %i total transformed' % (i, nCDP))

    print(stftWinLen)
    print(stftWinOver)

    if 'spectrogram' in features:
        outFeatures['freqs'] = freqs
        outFeatures['stftTimes'] = stftTimes
        outFeatures['spectrogram'] = np.abs(outFeatures['spectrogram'])

    return outFeatures
