# Stair-Climbing-Descending-Analysis-using-Silhouettes

## Thesis Code - Motion Heatmap and Machine Learning for Stair Climbing Detection
This code repository contains the code used to generate the results presented in my thesis titled "Motion Heatmap and Machine Learning for Stair Climbing Detection." In this thesis, we present a dataset of video data that includes bounding boxes information and silhouette images, along with the methods used to process this data to detect human movements, trajectories over time, and the usage of each room in the home environment.

## Dataset
The video data consists of two types of information: bounding boxes information (returns 3Dbb, 3DCen, 2Dbb, and 2DCen) and the actual silhouette image. Each silhouette frame image corresponds to 3D bounding boxes, 3D centroids, a UNIX timestamp (in milliseconds), and the camera information. There are three cameras in the SPHERE platform, each with a unique identifier (name), namely living room, kitchen, and hallway. We have four days (2016-07-25~2016-07-28) of videos that can be used for analysis.

## Motion Heatmap
In this section, we create a motion heatmap for detecting human movements, trajectories over time, and observe the usage of each room. The source of building this heatmap is inspired by Intel’s GitHub repository, which proposed this use of OpenCV back in 2018.

## Machine Learning
n this section, we introduce some concepts of machine learning for the stair climbing model. Python is the software of our coding platform, NumPy, pandas, sklearn, matplotlib are utilized here.
