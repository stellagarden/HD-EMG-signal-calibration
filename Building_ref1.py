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

def butter_bandpass_filter(data, lowcut=20.0, highcut=400.0, fs=1/0.000488, order=4):
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


def main():
    #loading .mat files consist of 0,1,2,3,11,17,18,21,23,24,25 gestures
    gestures = load_mat_files("../data/ref1_subject1_session1/")
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    gestures[0]=gestures[0][[1,3,6,7,10,12,18,24,25,29]]

    for gesture in gestures:
        for one_try in gesture:
            for i in range(len(one_try[0])):
                # Preprocessing : Applying Fourth order butterworth band-pass filter (20-400Hz)
                one_try[0][i]=butter_bandpass_filter(one_try[0][i])
                # Segmentation (1) Construct windows
                windows=[]
                for j in range(len(one_try[0][i])):
                    if j!=0 and j%150 == 0:
                        windows.append(one_try[0][i][j-150:j])
                # Segmentation (2) Compute RMS for each window
                rms_windows=[]
                for window in windows:
                    rms_windows.append(np.sqrt(np.mean(window**2)))
                print(type(one_try[0][i]))
                one_try[0][i] = list(one_try[0][i])
                one_try[0][i] = one_try[0][i].tolist()
                
                print(type(one_try[0][i]))
                one_try[0][i]=rms_windows
                
                print(type(one_try[0][i]))
                print(len(one_try[0][i]))
                break
            break
        break
    

    print(len(gestures[0][0][0][0]))
                


main()