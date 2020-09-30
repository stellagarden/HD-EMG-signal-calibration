import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import glob
import time
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
PLOT_PRINT_PROCESSING = False
PRINT_TIME_CONSUMING = True

def load_mat_files(dataDir):
    if PRINT_TIME_CONSUMING: t_load_mat_files=time.time()
    pathname=dataDir + "/**/*.mat"
    files = glob.glob(pathname, recursive=True)
    sessions=dict()
    #In idle gesture, we just use 2,4,7,8,11,13,19,25,26,30th tries in order to match the number of datas
    for one_file in files:
        session_name=one_file.split("\\")[-2]
        if not session_name in sessions:
            if one_file[-5:]=="0.mat":
                sessions[session_name]=np.array([io.loadmat(one_file)['gestures'][[1,3,6,7,10,12,18,24,25,29]]])
            else: sessions[session_name]=np.array([io.loadmat(one_file)['gestures']])
            continue
        if one_file[-5:]=="0.mat":
            sessions[session_name]=np.append(sessions[session_name], [io.loadmat(one_file)['gestures'][[1,3,6,7,10,12,18,24,25,29]]], axis=0)
            continue
        sessions[session_name]=np.append(sessions[session_name], [io.loadmat(one_file)['gestures']], axis=0)
    if PRINT_TIME_CONSUMING: print("Loading mat files: %.2f" %(time.time()-t_load_mat_files))
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

def compute_RMS_gestures(gestures):
    RMS_gestures=np.zeros((gestures.shape[:-1]))
    for i_ges in range(gestures.shape[0]):
        for i_try in range(gestures.shape[1]):
            for i_ch in range(gestures.shape[2]):
                for i_win in range(gestures.shape[3]):
                    RMS_gestures[i_ges, i_try, i_ch, i_win]=compute_RMS(gestures[i_ges, i_try, i_ch, i_win])
    return RMS_gestures

def base_normalization(RMS_gestures):
    if PRINT_TIME_CONSUMING: t_base_normalization=time.time()
    # Compute mean value of each channel of idle gesture
    average_channel_idle_gesture=np.mean(np.mean(RMS_gestures[0], 2), 0)
    # Subtract above value from every channel
    if PRINT_TIME_CONSUMING: print("## base_normalization: %.2f" %(time.time()-t_base_normalization))
    return np.transpose(np.transpose(RMS_gestures,(0,1,3,2))-average_channel_idle_gesture,(0,1,3,2))

def extract_ACTIVE_window_i(RMS_gestures):
    if PRINT_TIME_CONSUMING: t_extract_ACTIVE_window_i=time.time()
    RMS_gestures=np.transpose(RMS_gestures,(0,1,3,2))
    ## Determine whether ACTIVE : Compute summarized RMS
    sum_RMSs=np.sum(RMS_gestures,3)
    thresholds=np.reshape(np.repeat(np.sum(sum_RMSs,2)/sum_RMSs.shape[2], RMS_gestures.shape[2], axis=1),sum_RMSs.shape)
    ## Determine whether ACTIVE : Determining & Selecting the longest contiguous sequences
    i_ACTIVE_windows=np.zeros((sum_RMSs.shape[:-1]+(2,))).tolist()
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
            i_ACTIVE_windows[i_ges][i_try][0]=MAX_start
            i_ACTIVE_windows[i_ges][i_try][1]=MAX_contiguous
    if PRINT_TIME_CONSUMING: print("## extract_ACTIVE_window_i: %.2f" %(time.time()-t_extract_ACTIVE_window_i))
    return np.array(i_ACTIVE_windows)

def medfilt(channel, kernel_size=3):
    filtered=np.zeros(len(channel))
    for i in range(len(channel)):
        if i-kernel_size//2 <0 or i+kernel_size//2 >=len(channel):
            continue
        filtered[i]=median([channel[j] for j in range(i-kernel_size//2, i+kernel_size//2+1)])
    return filtered

def ACTIVE_filter(i_ACTIVE_windows, gestures):
    # ACTIVE_filter : delete if the window is not ACTIVE
    if PRINT_TIME_CONSUMING: t_ACTIVE_filter=time.time()
    list_gestures=np.transpose(gestures, (0,1,3,2,4)).tolist()
    for i_ges in range(i_ACTIVE_windows.shape[0]):
        for i_try in range(i_ACTIVE_windows.shape[1]):
            del list_gestures[i_ges][i_try][:i_ACTIVE_windows[i_ges][i_try][0]]
            del list_gestures[i_ges][i_try][i_ACTIVE_windows[i_ges][i_try][0]+i_ACTIVE_windows[i_ges][i_try][1]:]
    if PRINT_TIME_CONSUMING: print("## ACTIVE_filter: %.2f" %(time.time()-t_ACTIVE_filter))
    return np.array(list_gestures)

def Repartition_N_Compute_RMS(ACTIVE_gestures, N=SEGMENT_N):
    if PRINT_TIME_CONSUMING: t_Repartition_N_Compute_RMS=time.time()
    # List all the data of each channel without partitioning into windows
    ACTIVE_N_gestures=[[[[] for i_ch in range(len(ACTIVE_gestures[0][0][0]))] for i_try in range(ACTIVE_gestures.shape[1])] for i_ges in range(ACTIVE_gestures.shape[0])]     # CONSTANT
    for i_ges in range(len(ACTIVE_gestures)):
        for i_try in range(len(ACTIVE_gestures[i_ges])):
            for i_seg in range(len(ACTIVE_gestures[i_ges][i_try])):
                for i_ch in range(len(ACTIVE_gestures[i_ges][i_try][i_seg])):
                    ACTIVE_N_gestures[i_ges][i_try][i_ch].extend(ACTIVE_gestures[i_ges][i_try][i_seg][i_ch])
    # Compute RMS in N large windows
    for i_ges in range(len(ACTIVE_N_gestures)):
        for i_try in range(len(ACTIVE_N_gestures[i_ges])):
            for i_ch in range(len(ACTIVE_N_gestures[i_ges][i_try])):
                RMSs=[]
                for i  in range(N):
                    RMSs.append(compute_RMS(ACTIVE_N_gestures[i_ges][i_try][i_ch][(len(ACTIVE_N_gestures[i_ges][i_try][i_ch])//N)*i:(len(ACTIVE_N_gestures[i_ges][i_try][i_ch])//N)*(i+1)]))
                ACTIVE_N_gestures[i_ges][i_try][i_ch]=np.array(RMSs)
            ACTIVE_N_gestures[i_ges][i_try]=np.array(ACTIVE_N_gestures[i_ges][i_try]).transpose()   # Change (4,10,168,N) -> (4,10,N,168)
    if PRINT_TIME_CONSUMING: print("## Repartition_N_Compute_RMS: %.2f" %(time.time()-t_Repartition_N_Compute_RMS))
    return np.array(ACTIVE_N_gestures)

def mean_normalization(ACTIVE_N_RMS_gestures):
    if PRINT_TIME_CONSUMING: t_mean_normalization=time.time()
    for i_ges in range(len(ACTIVE_N_RMS_gestures)):
        for i_try in range(len(ACTIVE_N_RMS_gestures[i_ges])):
            for i_Lwin in range(len(ACTIVE_N_RMS_gestures[i_ges][i_try])):
                delta=max(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])-min(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])
                Mean=np.mean(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin])
                ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin]=(ACTIVE_N_RMS_gestures[i_ges][i_try][i_Lwin]-Mean)/delta
    if PRINT_TIME_CONSUMING: print("## mean_normalization: %.2f" %(time.time()-t_mean_normalization))
    return ACTIVE_N_RMS_gestures

def construct_X_y(mean_normalized_RMS):
    if PRINT_TIME_CONSUMING: t_construct_X_y=time.time()
    X=np.reshape(mean_normalized_RMS, (mean_normalized_RMS.shape[0]*mean_normalized_RMS.shape[1]*mean_normalized_RMS.shape[2], mean_normalized_RMS.shape[3]))
    y=np.array([])
    for i_ges in range(mean_normalized_RMS.shape[0]):
        for i in range(mean_normalized_RMS.shape[1]):   # # of tries
            for j in range(mean_normalized_RMS.shape[2]):  # # of Larege windows
                y=np.append(y, [i_ges])
    if PRINT_TIME_CONSUMING: print("## construct_X_y: %.2f" %(time.time()-t_construct_X_y))
    return X, y

def plot_confusion_matrix(y_test, kinds, y_pred):
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=kinds, yticklabels=kinds)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.axis('auto')
    plt.show()

def check_segment_len(i_ACTIVE_windows):
    for i in range(len(i_ACTIVE_windows)):
        print("%d번째 gesture의 각 try의 segment 길이들 : " %i, end='')
        for j in range(len(i_ACTIVE_windows[i])):
            print(i_ACTIVE_windows[i][j][1], end=' ')
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
    if PRINT_TIME_CONSUMING: t_extract_X_y_for_one_session=time.time()
    # Especially for Ref1, data reshaping into one array
    gestures=np.zeros((pre_gestures.shape[0], pre_gestures.shape[1])).tolist()      #CONSTANT
    for i_ges in range(len(pre_gestures)):
        for i_try in range(len(pre_gestures[i_ges])):
            gestures[i_ges][i_try]=pre_gestures[i_ges][i_try][0].copy()
    gestures=np.array(gestures)

    # Signal Pre-processing & Construct windows
    ## Segmentation : Data processing : Discard_useless_data
    if PRINT_TIME_CONSUMING: t_Discard_useless_data=time.time()
    gestures=np.delete(gestures,np.s_[7:192:8],2)
    if PRINT_TIME_CONSUMING: print("# Discard_useless_data: %.2f" %(time.time()-t_Discard_useless_data))
    if PLOT_PRINT_PROCESSING: plot_ch(gestures, 3, 2, 50)
    ## Preprocessing : Apply_butterworth_band_pass_filter
    if PRINT_TIME_CONSUMING: t_Apply_butterworth_band_pass_filter=time.time()
    for i_ges in range(len(gestures)):
        for i_try in range(len(gestures[i_ges])):
            gestures[i_ges, i_try]=butter_bandpass_filter(gestures[i_ges, i_try])
    if PRINT_TIME_CONSUMING: print("# Apply_butterworth_band_pass_filter: %.2f" %(time.time()-t_Apply_butterworth_band_pass_filter))
    if PLOT_PRINT_PROCESSING: plot_ch(gestures, 3, 2, 50)
    ## Segmentation : Data processing : Divide_continuous_data_into_150_samples_window
    if PRINT_TIME_CONSUMING: t_Divide_continuous_data_into_150_samples_window=time.time()
    gestures=np.delete(gestures, np.s_[(gestures.shape[3]//WINDOW_SIZE)*WINDOW_SIZE:], 3)
    gestures=np.reshape(gestures,(gestures.shape[0], gestures.shape[1], gestures.shape[2], gestures.shape[3]//WINDOW_SIZE, WINDOW_SIZE))
    if PRINT_TIME_CONSUMING: print("# Divide_continuous_data_into_150_samples_window: %.2f" %(time.time()-t_Divide_continuous_data_into_150_samples_window))

    # Determine ACTIVE windows
    ## Segmentation : Compute_RMS
    if PRINT_TIME_CONSUMING: t_Compute_RMS=time.time()
    RMS_gestures=compute_RMS_gestures(gestures)
    if PRINT_TIME_CONSUMING: print("# Compute_RMS: %.2f" %(time.time()-t_Compute_RMS))
    ## Segmentation : Base normalization
    if PLOT_PRINT_PROCESSING: 
        plt.imshow(RMS_gestures[3,2], cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
        plt.show()
    RMS_gestures=base_normalization(RMS_gestures)
    if PLOT_PRINT_PROCESSING: 
        plt.imshow(RMS_gestures[3,2], cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
        plt.show()
    ## Segmentation : Median_filtering
    if PRINT_TIME_CONSUMING: t_Median_filtering=time.time()
    RMS_gestures=np.apply_along_axis(medfilt, 3, RMS_gestures)
    if PRINT_TIME_CONSUMING: print("# Median filtering: %.2f" %(time.time()-t_Median_filtering))
    if PLOT_PRINT_PROCESSING: 
        plt.imshow(RMS_gestures[3,2], cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
        plt.show()
    ## Segmentation : Dertermine which window is ACTIVE
    i_ACTIVE_windows=extract_ACTIVE_window_i(RMS_gestures)

    # Feature extraction : Filter only ACTIVE windows
    ACTIVE_gestures=ACTIVE_filter(i_ACTIVE_windows, gestures)
    # Feature extraction : Partition existing windows into N large windows and compute RMS for each large window
    ACTIVE_N_RMS_gestures=Repartition_N_Compute_RMS(ACTIVE_gestures, SEGMENT_N)
    # Feature extraction : Mean normalization for all channels in each window
    mean_normalized_RMS=mean_normalization(ACTIVE_N_RMS_gestures)
    
    # Plot one data
    if PLOT_RANDOM_DATA:
        plot_some_data(mean_normalized_RMS)

    # Naive Bayes classifier : Construct X and y
    X, y = construct_X_y(mean_normalized_RMS)
    if PRINT_TIME_CONSUMING: print("extract_X_y_for_one_session: %.2f" %(time.time()-t_extract_X_y_for_one_session))
    return X, y

def plot_ch(data,i_gest,i_try=5,i_ch=89):
    plt.plot(data[i_gest][i_try][i_ch,:])
    plt.show()

def gnb_classifier(X, y, TEST_RATIO=TEST_RATIO):
    gnb = GaussianNB()
    if PRINT_TIME_CONSUMING: t_Training=time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=0)
    if PRINT_TIME_CONSUMING: print("Training: %.2f" %(time.time()-t_Training))
    if PRINT_TIME_CONSUMING: t_Testing=time.time()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    if PRINT_TIME_CONSUMING: print("Testing: %.2f" %(time.time()-t_Testing))
    print("Accuracy : %d%%" % (100-(((y_test != y_pred).sum()/X_test.shape[0])*100)))
    if PLOT_CONFUSION_MATRIX:
        plot_confusion_matrix(y_test, list(set(y)), y_pred)

def gmm_classifier(X, y, TEST_RATIO=TEST_RATIO):
    

def main():
    if PRINT_TIME_CONSUMING: t_main=time.time()
    sessions=load_mat_files("./data/")  # ndarray : sessions
    init_session=1
    for session in sessions.values():
        # Input data for each session
        X_session, y_session=extract_X_y_for_one_session(session)
        if init_session:
            X=np.array(X_session)
            y=np.array(y_session)
            init_session=0
            continue
        X=np.append(X, X_session, axis=0)
        y=np.append(y, y_session)

    # Naive Bayes classifier : Basic method : NOT LOOCV
    gnb_classifier(X, y)
    gmm_classifier(X, y)
    if PRINT_TIME_CONSUMING: print("main: %.2f" %(time.time()-t_main))

main()