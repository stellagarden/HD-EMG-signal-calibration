import time
t_construct_X_y=time.time()
import seaborn as sns
import random
import pandas as pd
import glob
from scipy.interpolate import interp2d
from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
print("## construct_X_y: %.2f" %(time.time()-t_construct_X_y))