from scipy.signal import butter, lfilter, freqz, medfilt
import numpy as np
from statistics import median

def medfilt(channel, kernel_size=3):
    filtered=np.zeros(len(channel))
    for i in range(len(channel)):
        if i-kernel_size//2 <0 or i+kernel_size//2 >=len(channel):
            continue
        filtered[i]=median([channel[j] for j in range(i-kernel_size//2, i+kernel_size//2+1)])
    return filtered

print(medfilt([8,52,1,546,12,15,48,1,32,4,84,5,21,5,9,4,23,4]))

# 0 1 2 3 4 5 6 7 8 9