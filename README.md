# 2020_Summer_Individual_study

## Notion

| .             | .                                                                                    |
| ------------- | ------------------------------------------------------------------------------------ |
| Notion Link   | [Individual SubNote](https://www.notion.so/SubNote-c44b5edc2bce4f158651a44a88177dc6) |
| Consists of   | Ref 1 summary, Ref 3 summary (lite), Programming timeline                            |
| Attached here | Programming tineline (2020.08.11 ver.)                                               |

<br>

## Data processing method
############################# NEED TO BE RECTIED ##################################   
Before processing, the RMS data is organized like the picture below.
![Before](/results/illust-data_structure.png)
There are various number of segments in each try, since active time is different for every try. Therefore, we'll make some **groups** for segments first. Suppose that we set N=n and there are M segments in some try. This means we want to make **n groups with M segments**.   
From 0th to (M/n\*1)-1th segments will grouped into one group. Similarly, from (M/n\*1)th to (M/n\*2)-1th segments will grouped, and so on. The remainings are ignored.   
When grouping is completed, compute RMS for each channel in each group. This results n numbers of 168-dimensional vectors in each group. In other words, we extracted **n numbers of data** that can be input into classifier.    
