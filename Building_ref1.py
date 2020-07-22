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
    init=1
    for window in windows:
        if init==1:
            RMSs=np.array([[compute_RMS(window)]])
            init=0
            continue
        RMSs=np.append(RMSs, [[compute_RMS(window)]], axis=0)
    return RMSs

def create_168_dimensional_window_vectors(one_try):
    for i in range(len(one_try[0])):
        if (i-1)%8 == 0:
            continue
        # Preprocessing : Applying Fourth order butterworth band-pass filter (20-400Hz)
        filtered_channel=butter_bandpass_filter(one_try[0][i])
        # Segmentation (1) Construct windows
        windows_per_channel=divide_to_windows(filtered_channel)
        # Segmentation (2) Compute RMS for each windows
        RMSwindows_per_channel=compute_RMS_for_each_windows(windows_per_channel)
        if i==0:
            RMS_one_try=np.array(RMSwindows_per_channel)
            continue
        RMS_one_try=np.append(RMS_one_try, RMSwindows_per_channel, axis=1)
    return RMS_one_try


def main():
    #loading .mat files consist of 0,1,2,3,11,17,18,21,23,24,25 gestures
    gestures = load_mat_files("../data/ref1_subject1_session1/")
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    gestures[0]=gestures[0][[1,3,6,7,10,12,18,24,25,29]]

    init_gesture=1
    for gesture in gestures:
        init_try=1
        for one_try in gesture:
            RMS_one_try = create_168_dimensional_window_vectors(one_try)
            if init_try == 1:
                RMS_tries_for_gesture = np.array([RMS_one_try])
                init_try=0
                continue
            RMS_tries_for_gesture = np.append(RMS_tries_for_gesture, [RMS_one_try], axis=0)

        if init_gesture==1:
            RMS_gestures = np.array([RMS_tries_for_gesture])
            init_gesture=0
            continue
        RMS_gestures = np.append(RMS_gestures, [RMS_tries_for_gesture], axis=0)
    
    print(len(RMS_gestures))
    print(len(RMS_gestures[0]))
    print(len(RMS_gestures[0][0]))
    print(len(RMS_gestures[0][0][0]))


main()