# Stair-Climbing-Descending-Analysis-using-Silhouettes

## Thesis Code - Motion Heatmap and Machine Learning for Stair Climbing Detection
This code repository contains the code used to generate the results presented in my thesis titled "Motion Heatmap and Machine Learning for Stair Climbing Detection." In this thesis, we present a dataset of video data that includes bounding boxes information and silhouette images, along with the methods used to process this data to detect human movements, trajectories over time, and the usage of each room in the home environment.

## Dataset
The video data consists of two types of information: bounding boxes information (returns 3Dbb, 3DCen, 2Dbb, and 2DCen) and the actual silhouette image. Each silhouette frame image corresponds to 3D bounding boxes, 3D centroids, a UNIX timestamp (in milliseconds), and the camera information. There are three cameras in the SPHERE platform, each with a unique identifier (name), namely living room, kitchen, and hallway. We have four days (2016-07-25~2016-07-28) of videos that can be used for analysis.

## Motion Heatmap
In this section, we create a motion heatmap for detecting human movements, trajectories over time, and observe the usage of each room. The source of building this heatmap is inspired by Intel’s GitHub repository, which proposed this use of OpenCV back in 2018.

## Machine Learning
In this section, we introduce some concepts of machine learning for the stair climbing model. Python is the software of our coding platform, NumPy, pandas, sklearn, matplotlib are utilized here.

# Linear regression
Here, we tried several machine learning algorithms. The one with the highest accuracy and the best performance for the stair climbing model would be selected and shown in the result. Linear regression is the simplest yet most primitive model of machine learning. It is used to predict real values based on continuous variables (land price, income, number of people, etc.). It is presented by a linear equation: 
𝑦 = 𝑚 ∗ 𝑥 + 𝑏
where y is the dependent variable, x is the independent variable, whereas m is the slope and b is the intercept. Scientist often use this equation, which is a straight line (called regression line) to predict the value of y for a given value of x.
Here, we use the linear regression as an algorithm for our stair climbing model. The accuracy (0.168%) and performance are both extremely low, this may be due to the inflexibility of linear regression and the fact that each 3Dcen coordinate is so far away from the other that it is difficult to find the best “straight line” for a linear regression model that is close to each point. Figure 23 shows the confusion matrix of linear regression. Showing 0 result of true negative and true positive.
On the other hand, we plot 3Dcen on 3D coordinates to visualize the performance of linear regression model. We set different colors for true positive (blue), true negative (yellow), false positive (green), false negative (black). 
![image](https://user-images.githubusercontent.com/117455557/225656752-d1e657b0-8c2b-48fb-9a21-f17df75e643e.png)
