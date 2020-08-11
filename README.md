# 2020_Summer_Individual_study [Working on ReadMe]

## Notion
Link : [Individual SubNote](https://www.notion.so/SubNote-c44b5edc2bce4f158651a44a88177dc6)
### Programming Timeline (2020.08.11)

Order,category,To-do,progress,Start,Due,Done,Done 1
1,Signal Preprocessing,Apply butterworth band-pass filter,Done,,,2주차,"Jul 21, 2020"
2,"Data processing, Segmentation",Divide continuous data into 150 samples window,Done,,,2주차,"Jul 21, 2020"
3,"Data processing, Segmentation",Discard useless data : 192ch → 168ch,Done,,,2주차,"Jul 22, 2020"
4,"Data processing, Segmentation",Compute RMS for each channel,Done,,,2주차,"Jul 22, 2020"
5,"Data processing, Segmentation",Perform baseline normalization,Done,"Jul 23, 2020",2주차,2주차,"Jul 24, 2020"
6,"Data processing, Segmentation",Check whether each window is represented by a 168-dimensional vector of RMS values,Done,"Jul 23, 2020",2주차,2주차,"Jul 24, 2020"
7,"Data processing, Segmentation",Apply spatial order 3 1-dimensional median filter on the vector to compensate local artifacts,Done,"Jul 28, 2020",3주차,3주차,"Jul 28, 2020"
8,"Determine whether ACTIVE, Segmentation",Compute average of the summarized RMS values per window → threshold,Done,"Jul 28, 2020",3주차,3주차,"Jul 28, 2020"
9,"Determine whether ACTIVE, Segmentation","If the sum of RMS vector elements of one window is greater than the threshold, it's ACTIVE",Done,"Jul 28, 2020",3주차,3주차,"Jul 28, 2020"
10,"Determine whether ACTIVE, Segmentation","If the predecessor and successor is active, it's ACTIVE",Done,"Jul 28, 2020",3주차,3주차,"Jul 28, 2020"
,Debugging,Check whether it's well operating until now,Done,"Jul 28, 2020",3주차,3주차,"Aug 1, 2020"
,Segmentation,Select the longest contiguous sequence of active windows → gesture segment,Done,"Jul 30, 2020",3주차,3주차,"Aug 1, 2020"
11,Feature Extraction,compute RMS for each channel on all windows → feature (of each channel),Done,"Aug 3, 2020",3주차,4주차,"Aug 3, 2020"
,,Edit code to be more effective (decrease the number of for loops),Quit,"Aug 3, 2020",,,
,,Apply feedbacks from mento,Done,"Aug 3, 2020",,4주차,"Aug 4, 2020"
12,Feature Extraction,Normalize the mean RMS over all channels,Done,"Aug 4, 2020",3주차,4주차,"Aug 4, 2020"
13,Feature Extraction,Result : 168 * N dimensional feature RMS vector. With RMS is length normalized,Done,"Aug 4, 2020",3주차,4주차,"Aug 4, 2020"
14,Naive Bayes classifier,Model the feature distribution by kernel density estimation with Gaussian kernel function,Done,"Aug 5, 2020",3주차,4주차,"Aug 6, 2020"
15,Naive Bayes classifier,Apply naive Bayes classifier for each 27 classes,Quit,,3주차,,
,,,,,,,
,,,,,,,
16,"Estimation of Electrode Displacement, Ulna position",Apply penalty function to favor the region in the middle of the array's x range,,,4주차,,
17,"Estimation of Electrode Displacement, Ulna position",Apply Watershed algorithm in order to find possible paths,Not understanded,,4주차,,
18,"Estimation of Electrode Displacement, Ulna position",Apply Dijkstra's algorithm to choose the lowest cost path,,,4주차,,
,,,,,,,
19,"Center of main muscle activity, Estimation of Electrode Displacement",Apply Gaussain Mixture Model (GMM),,,4주차,,
20,"Center of main muscle activity, Estimation of Electrode Displacement",Take mean of two estimation shift,,,4주차,,
,,,,,,,

- - -

## Classifying method
  There are two methods to handle data in order to input to classifier. Before handling, the data is organized like the picture below. They'll be converted to 2-dimensional matrix.
![Before](/pictures/illust-data_structure.png)

### Method 1
This method follows the reference 1.
> There will be various number of segments in each try, since active time will be different for every try. 

Calculate RMS values in each try for each channel. 
> Ex) We have "2 segments" consist of "m, n active windows", and "N=3".    
> Each window is consists of 168-dimensional vector. Therefore, if we process the segments in order to input into the classifier, X will be made like below.   
   
### Method 2
Flatten all the ACTIVE windows. Then active window becomes row of input data of classifier while columns are channels.

