# 2020_Summer_Individual_study [Working on ReadMe]

## Notion

| .             | .                                                                                    |
| ------------- | ------------------------------------------------------------------------------------ |
| Notion Link   | [Individual SubNote](https://www.notion.so/SubNote-c44b5edc2bce4f158651a44a88177dc6) |
| Consists of   | Ref 1 summary, Ref 3 summary (lite), Programming timeline                            |
| Attached here | Programming tineline (2020.08.11 ver.)                                               |

- - -

## Classifying method
  There are two methods to handle data in order to input to classifier. Before handling, the data is organized like the picture below. They'll be converted to 2-dimensional matrix.
![Before](/pictures/illust-data_structure.png)

### Method 1
This method follows the reference 1.
> There will be various number of segments in each try, since active time will be different for every try. First, we'll make some groups of segments. Let there are M segments in this try, and we set N=n.

Calculate RMS values in each try for each channel. 
> Ex) We have "2 segments" consist of "m, n active windows", and "N=3".    
> Each window is consists of 168-dimensional vector. Therefore, if we process the segments in order to input into the classifier, X will be made like below.   
   
### Method 2
Flatten all the ACTIVE windows. Then active window becomes row of input data of classifier while columns are channels.

