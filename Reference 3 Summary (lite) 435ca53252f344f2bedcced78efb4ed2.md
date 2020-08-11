# Reference 3 Summary (lite)

[High-Density Myoelectric Pattern Recognition Toward Improved Stroke Rehabilitation - IEEE Journals & Magazine](https://ieeexplore.ieee.org/abstract/document/6172561)

# Abstract

Myoelectric pattern-recognition techniques는 사람이 어떤 행동을 하려고 하는지 유추하는 분야에서 많이 발전되었다. 따라서 stroke 환자의 rehabilitation을 증진시키기 위해 EMG를 이용할 수 있을 것이다. 우리 연구에서는 high-density surface EMG를 이용하여 팔이 움직이려는 의도를 측정하고자 했다. 89개의 channel이 있는 surface EMG로 12분의 hemiparetic stroke subject를 대상으로 20가지의 서로 다른 팔과 손, 손가락/엄지의 움직임을 분석했다. 그 결과 매우 높은 classification accuracies (96.1% ± 4.3%)가 측정되었고 이는 마비된 환자들이 모터의 도움으로 움직일 수 있을 것이라는 가능성을 보여준다.

# Introduction

Stroke 환자의 일상생활 quality의 향상을 위해서라도 upper limb의 기능이 제대로 작동하지 않는 사람들은 해당 부분을 개선할 필요가 있다.

현재까지의 연구로 인해 수동적으로 움직이는 것 조차도 해당 부분과 뇌의 기능에 향상을 준다는 것이 밝혀졌다. 그런데 환자가 움직이려고 의도하고 이에 맞게 움직이는 active한 상황이 연출된다면 이는 치료에 훨씬 도움이 될 것이다. 

이 연구에서는 EMG를 신박한 방법으로 분석하여 subject가 어떤 행동을 하려고 의도 했는지 판단하기에 매우 높은 accuracies를 자랑한다. 이는 HD EMG를 이용한 점과 pattern-recognition analyses를 이용했기 때문이라 생각한다. 게다가 높은 수준의 accuracy를 계속 유지할 수 있었던 이유는 HD EMG의 모든 electrode를 사용하지 않고, 특정 electrode만 선별했기 때문이라 생각한다. 

# Methods

## A. Subjects

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled.png)

## B. Data Acquisition

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%201.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%201.png)

## C. Experimental Protocol

건강한 사람이 해당 동작을 하는 비디오를 보여줘서 가이드하였다. 각 동작 당 5번 씩 반복했다. 이때 3초 간 해당 동작을 하려고 유지하고 10초간 쉬고를 반복하면서 진행했다. 각 동작 사이의 휴식은 subject를 위해서 3-5분의 시간을 주었다.

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%202.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%202.png)

## D. Data Preprocessing and Segmentation

먼저 high-frequency noises와 movement artifact를 없애기 위해 fourth-order Butterworth band pass filter (30-500Hz)를 이용해서 preprocessing을 한다. Segmentation은 일반적으로 thresholding algorithm을 이용하지만 몇몇 session에서 subject들이 중립의 상태로 돌아오는 muscle activity도 기록되어 있기에 mislabeling을 피하기 위해 우리는 manual segmentation을 했다.

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%203.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%203.png)

각 segment를 길이가 256ms이고 128ms만큼 overlapping되는 window로 또 다시 segmentation하였고 이 window에 대해서 이후에 feature extraction을 하고 pattern classification method를 적용했다. Overlapped window를 사용한 이유는 제한된 data를 최대한 활용하고 classifier의 output이 연속되도록 만들기 위함이었다.

## E. Feature Extraction

EMG data를 characterize하기 위해 각 window에서 feature를 extract했다. 우리는 2가지 feature sets을 이용했다. 첫 번째는 time domain (TD) feature set이고, 두 번째는 autoregressive (AR) mode coefficients와 신호의 RMS 값의 combination이다. Feature set은 89개의 각 channel에 대해 계산되었고 이는 feature vector로 만들어졌다.

## F. Feature Dimensionality Reduction

위에서 생성한 feature들은 매우 높은 차원의 vector이다. TD feature set은 356-dimensional, AR+RMS feature set은 623-dimensional feature vectors이다. 따라서 높은 dimension을 줄이기 위해 principal component analysis (PCA)와 Fisher linear discriminant (FLD)의 활용형인 enhanced FLD model (EFM)을 이용하였다.

## G. Classification

우리는 3가지의 classification method를 이용했다.

1. MAP rule과 Bayesian principles를 이용한 Linear discriminant classifier (LDC)
2. [11]에서 제시된 것과 같은 GMM
3. Support vector machine (SVM). SVM은 우리와 같이 data samples가 제한적이고 높은 차원이고 nonlinear할 때 유용하게 이용된다.

## H. Performance Evaluation and Statistical Analysis

Pattern recognition은 각 subject에 대해 행해졌고 cross-validation scheme을 실행했다. 5번의 cross-validation의 averaged accuracy를 각 subject의 performance로 잡았고 전체 performance는 모든 subject에 대한 performance의 분포의 mean과 standard deviation으로 설정했다.

Paired t-tests were used to compare the pattern-recognition performance using different feature sets (TD and AR+RMS) or classifiers (LDC, GMM and SVM).

## I. Channel Selection

간단한 sequential feedforward selection algorithm이 사용됐다. 즉 classification accuracy에 가장 많이 반복적으로 유용한 정보를 제공한 channel이 선택되었다.

# Experimental Results

## A. Characterization of Muscle Activity Patterns

각 channel의 RMS value로 contour plot을 그려서 characterization했다.

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%204.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%204.png)

## B. Effect of Number of PCs on Performance

아래의 결과에 따라 TD feature set일 때는 m=150, AR+RMS일 때는 m=220을 갖는 것이 가장 optimal하다.

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%205.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%205.png)

## C. Classification of Intended Movements in Stroke

TD가 AR+RMS보다 전체적으로 좋은 결과가 나왔는데 이는 더 적은 feature dimension을 가지기 때문이라 추측한다. Classifier는 GMM과 SVM이 LDC 보다 더 좋은 정확도를 냈고, 종합적으로는 TD와 SVM의 조합이 가장 좋았다.

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%206.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%206.png)

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%207.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%207.png)

## D. Preliminary Channel Selection Analysis

EMG channel이 8개 여도 충분하다.

어떤 규칙으로 EMG channel을 선택한 것인가?

![Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%208.png](Reference%203%20Summary%20(lite)%20435ca53252f344f2bedcced78efb4ed2/Untitled%208.png)

# Discussions and Conclusion

본 연구에서는 high-density surface EMG를 분석할 때 pattern-recognition techniques를 적용하는 것이 accuracy를 더 높여준다는 것을 증명했다.