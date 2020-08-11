# 2020_Summer_Individual_study [Working on ReadMe]

## Notion

| .             | .                                                                                    |
| ------------- | ------------------------------------------------------------------------------------ |
| Notion Link   | [Individual SubNote](https://www.notion.so/SubNote-c44b5edc2bce4f158651a44a88177dc6) |
| Consists of   | Ref 1 summary, Ref 3 summary (lite), Programming timeline                            |
| Attached here | Programming tineline (2020.08.11 ver.)                                               |

<br>

## Classifying method
  There are two methods to handle data in order to input to classifier. Before handling, the data is organized like the picture below. They'll be converted to 2-dimensional matrix.
![Before](/pictures/illust-data_structure.png)

### Method 1
This method follows the reference 1.   
There are various number of segments in each try, since active time is different for every try. Therefore, we'll make some **groups** for segments first. Suppose that we set N=n and there are M segments in some try. This means we want to make n groups with M segments.
From **0**th to **(M/n\*1)-1**th segments will grouped into one group. Similarly, from **(M/n\*1)** th to **(M/n\*2)-1**th segments will grouped, and so on. The remainings are ignored.   
When grouping is completed, compute RMS for each channel in each group. This results n numbers of 168-dimensional vectors in each group.   
If we flatten the n vectors and attach the tries vertically continously, 2-dimensional matrix will be constructed.   
**--Add final data structure illust--**

### Method 2
Just attach all the segments and tries vertically continously, so that one segment is regarded as one data in classifier.   
**--Add final data structure illust--**
