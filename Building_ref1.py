import os
from scipy import io

def load_mat_files(dataDir):
    mats = []
    for file in os.listdir(dataDir):
        mats.append(io.loadmat(dataDir+file)['gestures'])
        
    return mats


def main():
    mats = load_mat_files("../data/ref1_subject1_session1/")


main()