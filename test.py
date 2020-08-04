from scipy.signal import butter, lfilter, freqz, medfilt
import numpy as np
a=np.array([1,2,36,1,8,9,2,4,6,2,0,12,5,0])
print(a)
print(medfilt(a, kernel_size=3))

# 0 1 2 3 4 5 6 7 8 9