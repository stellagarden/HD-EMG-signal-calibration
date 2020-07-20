import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.signal import butter, lfilter, freqz

def load_mat_files(dataDir):
    mats = []
    for file in os.listdir(dataDir):
        mats.append(io.loadmat(dataDir+file)['gestures'])

    return mats

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def main():
    #loading .mat files consist of 0,1,2,3,11,17,18,21,23,24,25 gestures
    gestures = load_mat_files("../data/ref1_subject1_session1/")
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    gestures[0]=gestures[0][[1,3,6,7,10,12,18,24,25,29]]
    print(len(gestures[0]))
"""
    #Preprocessing : Applying Fourth order butterworth band-pass filter (20-400Hz)
    fs = 5000.0
    lowcut = 20.0
    highcut = 400.0
    T = 0.05
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    plt.figure(1)
    plt.clf()
    plt.plot(t, gestures[0], label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()"""

main()