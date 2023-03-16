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
The confusion matrix of linear regression  
![image](https://user-images.githubusercontent.com/117455557/225656752-d1e657b0-8c2b-48fb-9a21-f17df75e643e.png)  
The 3Dcen scatter plot of linear regression model  
![image](https://user-images.githubusercontent.com/117455557/225656938-fa58483b-9317-4777-9353-69e3d66765fa.png)  
As shown in the above figure, linear regression model does not correctly predict stair climbing and other motions.

# K-Nearest Neighbor (KNN)  
The KNN method is a basic, supervised machine learning algorithm which can be used to solve regression and classification problems. The letter “K” means how many neighbors to consider near the target point. Given a test instance (i), the KNN algorithm would find the k nearest neighbors and their labels, the label of i would be predicted as the majority of the labels of the k nearest neighbors.
The main area of application of KNN algorithms is the classification of unknowns; determining which category does the unknown belong to. It compares the training data to the new point, returning the most frequent class of the K nearest points. However, choosing a good K allows the trained model to be flexible enough to avoid overfitting and underfitting. Here, we split the entire data into 3 parts, that is, the training set (60% of the total data), testing set (20%), and cross-validation set (20%). The training data is used to find the nearest neighbors, whereas the cross-validation data is used to find the best number of neighbors to consider (value of K). We then test the KNN model on the testing dataset and obtained the below figure. The cross-validation shows the highest accuracy at K = 27, therefore, we consider 27 neighbors in the KNN model.  
The cross-validation accuracy of KNN  
![image](https://user-images.githubusercontent.com/117455557/225657792-cfcfd5a6-a417-4e88-8bdb-c87d846c882a.png)  
The accuracy is 98.2%, whereas the precision is 74.3%, recall is 25.1% and the F1 score is 67.7%. The ROC AUC(area under the curve) score is 74.4%, which shows the true positive rate is higher than the false positive rate. Overall, the performance of the KNN model is good. In the scatter plot model, KNN successfully predicts the stair climbing (blue) and non-stair climbing (yellow) motions. There is some false negative (black) gathering in the front door, but all in all the model performs well and we, therefore, decide to select KNN as the stair detection model algorithm.  
The confusion matrix of KNN  
![image](https://user-images.githubusercontent.com/117455557/225657911-096045b4-b34b-4971-8a97-2ab76fb5bfe2.png)  
The 3Dcen scatter plot of KNN model  
![image](https://user-images.githubusercontent.com/117455557/225657952-9d1521d0-ef63-47d9-a90b-78843da56572.png)  

