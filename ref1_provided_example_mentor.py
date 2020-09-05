# example code to load data from the csl-hdemg dataset

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from math import sqrt

from sklearn.naive_bayes import GaussianNB

num_gest = 27
X = np.zeros((270, 168))
y = np.zeros((270, 1))
#plot data
fig, axes = plt.subplots(nrows=5, ncols=6)
gnb = GaussianNB()
for gest in range(num_gest):
    print("Processing... %s"%(str(gest)))
    mat = sio.loadmat('../subject1/session5/gest'+str(gest)+'.mat')
    gestures = mat['gestures']

    #compute RMS
    for i in range(0,10):
        rms = np.zeros(168)
        trial = gestures[i,0]
        #deleting edge channels
        trial = np.delete(trial,np.s_[7:192:8],0)
        for c in range(0,trial.shape[0]):
            #computing mean rms over all repetitions
            rms[c] += np.linalg.norm(trial[c,:]) / sqrt(len(trial[c,:]))
        X[gest*10+i, :] = rms
        y[gest*10+i, :] = gest

    #reshaping to the correct shape

    rms = np.reshape(rms,(24,7))
    rms = np.flipud(np.transpose(rms))

    axes[gest%5,gest%6].imshow(rms, cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.0035)
    axes[gest%5,gest%6].set_title(str(gest))

plt.show()


from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy : %d%%" % (100-(((y_test.flatten() != y_pred).sum()/X_test.shape[0])*100)))

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.axis('auto')
plt.show()
