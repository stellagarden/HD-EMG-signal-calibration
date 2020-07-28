import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.signal import butter, lfilter, freqz, medfilt

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
        # Segmentation : Data processing : Discard uesless data
        if (i-1)%8 == 0:
            continue
        # Preprocessing : Apply butterworth band-pass filter
        filtered_channel=butter_bandpass_filter(one_try[0][i])
        # Segmentation : Data processing : Divide continuous data into 150 samples window
        windows_per_channel=divide_to_windows(filtered_channel)
        # Segmentation : Compute RMS for each channel
        RMSwindows_per_channel=compute_RMS_for_each_windows(windows_per_channel)
        if i==0:
            RMS_one_try=np.array(RMSwindows_per_channel)
            continue
        RMS_one_try=np.append(RMS_one_try, RMSwindows_per_channel, axis=1)
    return RMS_one_try

def average_for_channel(gesture):
    average=np.array([])
    for i_ch in range(gesture.shape[2]):
        sum=0
        for i_win in range(gesture.shape[1]):
            for i_try in range(gesture.shape[0]):
                sum+=gesture[i_try][i_win][i_ch]
        average=np.append(average, [sum])
    return average

def base_normalization(RMS_gestures):
    average_channel_idle_gesture=average_for_channel(RMS_gestures[0])
    print(average_channel_idle_gesture[:4])
    for i_ges in range(RMS_gestures.shape[0]):
        for i_try in range(RMS_gestures.shape[1]):
            for i_win in range(RMS_gestures.shape[2]):
                for i_ch in range(RMS_gestures.shape[3]):
                    RMS_gestures[i_ges][i_try][i_win][i_ch]-=average_channel_idle_gesture[i_ch]
    return RMS_gestures

def ACTIVE_filter(RMS_gestures):
    for i_ges in range(len(RMS_gestures)):
        for i_try in range(len(RMS_gestures[i_ges])):
            # Segmentation : Determine whether ACTIVE : Compute summarized RMS
            sum_RMSs=[sum(window) for window in RMS_gestures[i_ges][i_try]]
            threshold=sum(sum_RMSs)/len(sum_RMSs)
            # Segmentation : Determine whether ACTIVE
            i_ACTIVEs=[]
            for i_win in range(len(RMS_gestures[i_ges][i_try])):
                if sum_RMSs[i_win] > threshold:
                    i_ACTIVEs.append(i_win)
            for i in range(len(i_ACTIVEs)):
                if i==0:
                    continue
                if i_ACTIVEs[i]-i_ACTIVEs[i-1] == 2:
                    i_ACTIVEs.insert(i, i_ACTIVEs[i-1]+1)
            # Segmentation : Determine whether ACTIVE : delete if the window is not ACTIVE
            for i_win in reversed(range(len(RMS_gestures[i_ges][i_try]))):
                if not i_win in i_ACTIVEs:
                    print(i_ges, i_try, i_win)
                    del RMS_gestures[i_ges][i_try][i_win]
    return RMS_gestures


def main():
    #loading .mat files consist of 0,1,2,3,11,17,18,21,23,24,25 gestures
    gestures = load_mat_files("../data/ref1_subject1_session1/")
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    gestures[0]=gestures[0][[1,3,6,7,10,12,18,24,25,29]]
    # Signal Preprocessing & Data processing for segmentation
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
    # Segmentation : Data processing : Base normalization
    RMS_gestures=base_normalization(RMS_gestures)
    # Segmentation : Data processing : Median filtering
    RMS_gestures=medfilt(RMS_gestures, kernel_size=3)
    # Segmentation : Dertermine which window is ACTIVE
    ACTIVE_RMS_gestures=ACTIVE_filter(RMS_gestures.tolist())

    print(len(ACTIVE_RMS_gestures))
    print(len(ACTIVE_RMS_gestures[0]))
    print(len(ACTIVE_RMS_gestures[0][0]))
    print(len(ACTIVE_RMS_gestures[0][0][0]))

main()