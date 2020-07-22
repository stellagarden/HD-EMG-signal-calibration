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

def butter_bandpass_filter(data, lowcut=20.0, highcut=400.0, fs=2048, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def plot_bandpass_filtered_data(data):
    plt.figure(1)
    plt.clf()
    plt.plot(data, label='Noisy signal')
 
    y = butter_bandpass_filter(data)
    plt.plot(y, label='Filtered signal')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()

def divide_to_windows(datas, window_size=150):
    windows=np.delete(datas, list(range((len(datas)//window_size)*window_size,len(datas))))
    windows=np.reshape(windows,((len(datas)//window_size,window_size)))
    return windows

def compute_RMS(datas):
    return np.sqrt(np.mean(datas**2))

def compute_RMS_for_each_windows(windows):
    RMSs=np.array([])
    for window in windows:
        RMSs=np.append(RMSs, compute_RMS(window))
    return RMSs


def main():
    #loading .mat files consist of 0,1,2,3,11,17,18,21,23,24,25 gestures
    gestures = load_mat_files("../data/ref1_subject1_session1/")
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    gestures[0]=gestures[0][[1,3,6,7,10,12,18,24,25,29]]

    for gesture in gestures:
        for one_try in gesture:
            for channel in one_try[0]:
                # Preprocessing : Applying Fourth order butterworth band-pass filter (20-400Hz)
                channel=butter_bandpass_filter(channel)
                # Segmentation (1) Construct windows
                windows=divide_to_windows(channel)
                # Segmentation (2) Compute RMS for each window
                RMSwindows=compute_RMS_for_each_windows(windows)
                


                break
            break
        break
                


main()