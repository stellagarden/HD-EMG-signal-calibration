# 2020_Summer_Individual_study
Will add comments later
## Classifying method 1
Calculate RMS values with each segments for each channels.
Ex) We have "2 segments" consist of "m, n active windows", and "N=3".
    Each window is consists of 168-dimensional vector. Therefore, if we process the segments in order to input into the classifier, X will be made like below.

## Classifiying method 2
Flatten all the ACTIVE windows. Then active window becomes row of input data of classifier while columns are channels.