import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.signal import butter, lfilter, freqz
from statistics import median
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X=np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
X2=np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2]))
print(X2)