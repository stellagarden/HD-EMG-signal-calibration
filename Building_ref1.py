import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import glob
from mpl_toolkits import mplot3d
from scipy import io
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import interp2d
from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
WINDOW_SIZE = 150    # 20:9.76ms, 150:73.2ms
TEST_RATIO = 0.3
SEGMENT_N = 3
PLOT_RANDOM_DATA = False
PLOT_CONFUSION_MATRIX = True
ACTUAL_COLUMN=24
ACTUAL_RAW=7
IDLE_GESTURE_EXIST = True

def load_mat_files(dataDir):
    pathname=dataDir + "/**/*.mat"
    files = glob.glob(pathname, recursive=True)
    sessions=dict()
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    for one_file in files:
        session_name=one_file.split("\\")[-2]
        if not session_name in sessions:
            if one_file[-5:]=="0.mat" and IDLE_GESTURE_EXIST == True:
                sessions[session_name]=np.array([io.loadmat(one_file)['gestures'][[1,3,6,7,10,12,18,24,25,29]]])
            else: sessions[session_name]=np.array([io.loadmat(one_file)['gestures']])
            continue
        if one_file[-5:]=="0.mat" and IDLE_GESTURE_EXIST == True:
            sessions[session_name]=np.append(sessions[session_name], [io.loadmat(one_file)['gestures'][[1,3,6,7,10,12,18,24,25,29]]], axis=0)
            continue
        sessions[session_name]=np.append(sessions[session_name], [io.loadmat(one_file)['gestures']], axis=0)
    return sessions

def butter_bandpass_filter(data, lowcut=20.0, highcut=400.0, fs=2048, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def compute_RMS(datas):
    return np.sqrt(np.mean(np.array(datas)**2))

def base_normalization(RMS_gestures):
    # Compute mean value of each channel of idle gesture
    average_channel_idle_gesture=np.mean(np.mean(RMS_gestures[0], 2), 0)
    # Subtract above value from every channel
    return np.transpose(np.transpose(RMS_gestures,(0,1,3,2))-average_channel_idle_gesture,(0,1,3,2))

def extract_ACTIVE_window_i(RMS_gestures):
    RMS_gestures=np.transpose(RMS_gestures,(0,1,3,2))
    ## Determine whether ACTIVE : Compute summarized RMS
    sum_RMSs=np.sum(RMS_gestures,3)
    thresholds=np.reshape(np.repeat(np.sum(sum_RMSs,2)/sum_RMSs.shape[2], RMS_gestures.shape[2], axis=1),sum_RMSs.shape)
    ## Determine whether ACTIVE : Determining & Selecting the longest contiguous sequences
    i_ACTIVE_windows=np.zeros((sum_RMSs.shape[:-1]+(2,)))
    sum_RMSs=sum_RMSs-thresholds
    for i_ges in range(sum_RMSs.shape[0]):
        for i_try in range(sum_RMSs.shape[1]):
            contiguous = 0
            MAX_contiguous = 0
            for i_win in range(sum_RMSs.shape[2]):
                sandwitch=i_win!=0 and i_win!=sum_RMSs.shape[2]-1 and sum_RMSs[i_ges, i_try, i_win-1]>0 and sum_RMSs[i_ges, i_try, i_win+1]>0
                if sum_RMSs[i_ges, i_try, i_win]>0 or sandwitch:
                    if contiguous==0: i_start=i_win
                    contiguous+=1
                    if i_win!=sum_RMSs.shape[2]-1: continue
                if contiguous!=0:
                    if MAX_contiguous<contiguous:
                        MAX_start=i_start
                        MAX_contiguous=contiguous
                    else:
                        contiguous=0
            i_ACTIVE_windows[i_ges, i_try, 0]=MAX_start
            i_ACTIVE_windows[i_ges, i_try, 1]=MAX_contiguous
    return i_ACTIVE_windows

def medfilt(channel, kernel_size=3):
    filtered=np.zeros(len(channel))
    for i in range(len(channel)):
        if i-kernel_size//2 <0 or i+kernel_size//2 >=len(channel):
            continue
        filtered[i]=median([channel[j] for j in range(i-kernel_size//2, i+kernel_size//2+1)])
    return filtered

def ACTIVE_filter(i_ACTIVE_windows, pre_processed_gestures):
    # ACTIVE_filter : delete if the window is not ACTIVE
    list_pre_processed_gestures=pre_processed_gestures.tolist()
    for i_ges in range(len(list_pre_processed_gestures)):
        for i_try in range(len(list_pre_processed_gestures[i_ges])):
            for i_win in reversed(range(len(list_pre_processed_gestures[i_ges][i_try]))):
                if not i_win in range(i_ACTIVE_windows[i_ges][i_try][0], i_ACTIVE_windows[i_ges][i_try][0]+i_ACTIVE_windows[i_ges][i_try][1]):
                    del list_pre_processed_gestures[i_ges][i_try][i_win]
    return np.array(list_pre_processed_gestures)

def Repartition_N_Compute_RMS(ACTIVE_pre_processed_gestures, N=SEGMENT_N):
    # List all the data of each channel without partitioning into windows
    ACTIVE_N_gestures=[[[[] for i_ch in range(len(ACTIVE_pre_processed_gestures[0][0][0]))] for i_try in range(ACTIVE_pre_processed_gestures.shape[1])] for i_ges in range(ACTIVE_pre_processed_gestures.shape[0])]     # CONSTANT
    for i_ges in range(len(ACTIVE_pre_processed_gestures)):
        for i_try in range(len(ACTIVE_pre_processed_gestures[i_ges])):
            for i_seg in range(len(ACTIVE_pre_processed_gestures[i_ges][i_try])):
                for i_ch in range(len(ACTIVE_pre_processed_gestures[i_ges][i_try][i_seg])):
                    ACTIVE_N_gestures[i_ges][i_try][i_ch].extend(ACTIVE_pre_processed_gestures[i_ges][i_try][i_seg][i_ch])
    # Compute RMS in N large windows
    for i_ges in range(len(ACTIVE_N_gestures)):
        for i_try in range(len(ACTIVE_N_gestures[i_ges])):
            for i_ch in range(len(ACTIVE_N_gestures[i_ges][i_try])):
                RMSs=[]
                for i  in range(N):
                    RMSs.append(compute_RMS(ACTIVE_N_gestures[i_ges][i_try][i_ch][(len(ACTIVE_N_gestures[i_ges][i_try][i_ch])//N)*i:(len(ACTIVE_N_gestures[i_ges][i_try][i_ch])//N)*(i+1)]))
                ACTIVE_N_gestures[i_ges][i_try][i_ch]=np.array(RMSs)
            ACTIVE_N_gestures[i_ges][i_try]=np.array(ACTIVE_N_gestures[i_ges][i_try]).transpose()   # Change (4,10,168,N) -> (4,10,N,168)
    return np.array(ACTIVE_N_gestures)

def mean_normalization(ACTIVE_N_RMS_gestures):
    for i_ges in range(len(ACTIVE_N_RMS_gestures)):
        for i_try in range(len(ACTIVE_N_RMS_gestures[i_ges])):
            for i_Lwin in range(len(ACTIVE_N_RMS_gestures[i_ges][i_try])):
                delta=max(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])-min(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])
                Mean=np.mean(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])
                for i_ch in range(len(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])):
                    ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin][i_ch]=(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin][i_ch]-Mean)/delta
    return ACTIVE_N_RMS_gestures

def construct_X_y(mean_normalized_RMS):
    X=np.reshape(mean_normalized_RMS, (mean_normalized_RMS.shape[0]*mean_normalized_RMS.shape[1]*mean_normalized_RMS.shape[2], mean_normalized_RMS.shape[3]))
    y=np.array([])
    for i_ges in range(mean_normalized_RMS.shape[0]):
        for i in range(mean_normalized_RMS.shape[1]):   # # of tries
            for j in range(mean_normalized_RMS.shape[2]):  # # of Larege windows
                y=np.append(y, [i_ges])
    return X, y

def plot_confusion_matrix(y_test, kinds, y_pred):
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=kinds, yticklabels=kinds)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.axis('auto')
    plt.show()

def check_segment_len(ACTIVE_RMS_gestures):
    for i in range(len(ACTIVE_RMS_gestures)):
        print("%d번째 gesture의 각 try의 segment 길이들 : " %i, end='')
        for j in range(len(ACTIVE_RMS_gestures[i])):
            print(len(ACTIVE_RMS_gestures[i][j]), end=' ')
        print()

def plot_some_data(gestures):
    # Choose random three data
    chose=[]
    for i in range(3):
        rand_ges = random.randint(1, len(gestures)-1)    # Except idle gesture
        rand_try = random.randint(0, len(gestures[rand_ges])-1)
        rand_win = random.randint(0, len(gestures[rand_ges][rand_try])-1)
        chose.append((rand_ges, rand_try, rand_win))
    # Plot
    y,x=np.meshgrid(range(ACTUAL_RAW),range(ACTUAL_COLUMN))
    fig, ax = plt.subplots(nrows=3)
    im=[]
    for i in range(len(chose)):
        df = pd.DataFrame({"x":x.flatten(), "y":y.flatten(),"value":gestures[chose[i][0]][chose[i][1]][chose[i][2]].flatten()}).pivot(index="y", columns="x", values="value")
        im.append(ax[i].imshow(df.values, cmap="viridis", vmin=0, vmax=1))
        ax[i].set_title("%dth active window in %dth try in %dth gesture" %(chose[i][2], chose[i][1], chose[i][0]))
        fig.colorbar(im[i], ax=ax[i])
    plt.tight_layout()
    plt.show()

def extract_X_y_for_one_session(pre_gestures):
    # Especially for Ref1, data reshaping into one array
    gestures=np.zeros((pre_gestures.shape[0], pre_gestures.shape[1])).tolist()      #CONSTANT
    for i_ges in range(len(pre_gestures)):
        for i_try in range(len(pre_gestures[i_ges])):
            gestures[i_ges][i_try]=pre_gestures[i_ges][i_try][0].copy()
    gestures=np.array(gestures)

    # Signal Pre-processing & Construct windows
    ## Segmentation : Data processing : Discard useless data
    gestures=np.delete(gestures,np.s_[7:192:8],2)
    plot_ch(gestures, 3, 2, 50)
    ## Preprocessing : Apply butterworth band-pass filter
    gestures=np.apply_along_axis(butter_bandpass_filter, 2, gestures)
    plot_ch(gestures, 3, 2, 50)
    ## Segmentation : Data processing : Divide continuous data into 150 samples window
    gestures=np.delete(gestures, np.s_[(gestures.shape[3]//WINDOW_SIZE)*WINDOW_SIZE:], 3)
    gestures=np.reshape(gestures,(gestures.shape[0], gestures.shape[1], gestures.shape[2], gestures.shape[3]//WINDOW_SIZE, WINDOW_SIZE))
    
    # Determine ACTIVE windows
    ## Segmentation : Compute RMS
    RMS_gestures=gestures.copy()
    RMS_gestures=np.apply_along_axis(compute_RMS, 4, RMS_gestures)
    ## Segmentation : Base normalization
    plt.imshow(RMS_gestures[3,2], cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
    plt.show()
    RMS_gestures=base_normalization(RMS_gestures)
    plt.imshow(RMS_gestures[3,2], cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
    plt.show()
    ## Segmentation : Median filtering
    RMS_gestures=np.apply_along_axis(medfilt, 3, RMS_gestures)
    plt.imshow(RMS_gestures[3,2], cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
    plt.show()
    ## Segmentation : Dertermine which window is ACTIVE
    i_ACTIVE_windows=extract_ACTIVE_window_i(RMS_gestures)

    ########################### WORKING LINE #############################
    # Feature extraction : Filter only ACTIVE windows
    ACTIVE_pre_processed_gestures=ACTIVE_filter(i_ACTIVE_windows, gestures)
    # Feature extraction : Partition existing windows into N large windows and compute RMS for each large window
    ACTIVE_N_RMS_gestures=Repartition_N_Compute_RMS(ACTIVE_pre_processed_gestures, SEGMENT_N)
    # Feature extraction : Mean normalization for all channels in each window
    mean_normalized_RMS=mean_normalization(ACTIVE_N_RMS_gestures)
    
    # Plot one data
    if PLOT_RANDOM_DATA==True:
        plot_some_data(mean_normalized_RMS)

    # Naive Bayes classifier : Construct X and y
    X, y = construct_X_y(mean_normalized_RMS)
    return X, y

def plot_ch(data,i_gest,i_try=5,i_ch=89):
    plt.plot(data[i_gest][i_try][i_ch,:])
    plt.show()

def main():
    sessions=load_mat_files("./data/")  # Dict : sessions
    init_session=1
    for session in sessions.values():
        # Input data for each session
        X_session, y_session=extract_X_y_for_one_session(session)
        print("Processing...%d" %(sessions.values().index(session)))
        if init_session==1:
            X=np.array(X_session)
            y=np.array(y_session)
            init_session=0
            continue
        X=np.append(X, X_session, axis=0)
        y=np.append(y, y_session)
    kinds=list(set(y))

    # Naive Bayes classifier : Basic method : NOT LOOCV
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=0)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Accuracy : %d%%" % (100-(((y_test != y_pred).sum()/X_test.shape[0])*100)))
    if PLOT_CONFUSION_MATRIX:
        plot_confusion_matrix(y_test, kinds, y_pred)
    
main()