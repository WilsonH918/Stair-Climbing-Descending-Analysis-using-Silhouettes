# importing needed modules
import pandas as pd
from pandas.io import formats
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


########################################################################
############ DATA CLEANING AND PLOTTING THE ENVIRONMENT ################
########################################################################

data = pd.read_csv('stair_climbing_data.csv')

select_column = pd.DataFrame(data, columns=['dt', '3Dcen-1', '3Dcen-2', '3Dcen-3'])

# Select the time that people are doing "stair climbing"
select_time = select_column[(select_column['dt'] >= '2016-07-25T18:21:10.579Z') & 
                            (select_column['dt'] <= '2016-07-25T18:21:27.135Z') |
                            (select_column['dt'] >= '2016-07-26T00:11:51.626Z') & 
                            (select_column['dt'] <= '2016-07-26T00:12:14.250Z') |
                            (select_column['dt'] <= '2016-07-26T00:18:32.178Z') & 
                            (select_column['dt'] >= '2016-07-26T00:18:42.340Z')]
select_time2 = select_column[(select_column['dt'] > '2016-07-25T18:21:27.135Z')]

# Convert the data to numpy
xs = select_time['3Dcen-1'].to_numpy()
ys = select_time['3Dcen-2'].to_numpy()
zs = select_time['3Dcen-3'].to_numpy()

xs2 = select_time2['3Dcen-1'].to_numpy()
ys2 = select_time2['3Dcen-2'].to_numpy()
zs2 = select_time2['3Dcen-3'].to_numpy()

# Denoise the data, filter out data that doesn't make sense
new_x = np.where(xs<0, 500, xs)

# Data Cleaning; Putting the data into the right shape for future plots
xyz = np.vstack((xs,ys,zs))
xyz2 = np.vstack((xs2,ys2,zs2))
xyz_detected = xyz.T
xyz_nondetected = xyz2.T
xyz_total = np.vstack((xyz_detected,xyz_nondetected))


label_detected = np.ones(1971) #stair detection found
label_nondetected = np.zeros(96926) #others
label_total = np.append(label_detected,label_nondetected)

# 2d plot of the hallway environmet; We set stair climbing motions as blue dots, while others as yellow
def plot_2d():
    fig = plt.figure(figsize=(15,15))

    plt.plot(xs2,ys2,'y.')
    plt.plot(new_x,ys,'b.')    
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')  
    plt.title('stair detection')    
    plt.grid(True)    
    plt.show()

#plot_2d()

# 3d plot, 90 degrees of the hallway
def plot_3d_90():
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
        
    for j in range(len(xs2)):
        ax.scatter(xs2[j],ys2[j],zs2[j], color = 'yellow', marker = '.')
            
    for i in range(len(xs)):
        ax.scatter(new_x[i],ys[i],zs[i], color = 'blue', marker='.')
    


    ax.view_init(elev=90, azim=-90)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('D Label')
    plt.show()

#plot_3d_90()

# 3d plot, 45 degrees of the hallway
def plot_3d_45():
    fig = plt.figure(figsize=(15,15))

    ax = fig.add_subplot(projection='3d')
        
    for j in range(len(xs2)):
        ax.scatter(xs2[j],ys2[j],zs2[j], color = 'yellow', marker = '.')
            
    for i in range(len(xs)):
        ax.scatter(new_x[i],ys[i],zs[i], color = 'blue', marker='.')
    

    ax.view_init(elev=20, azim=-70)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('D Label')
    plt.show()

#plot_3d_45()

## Creating tools we are going to test our model later
# To test the sensitivity of the model
def Sensitivity_Test():
    TP = []
    FP = []
    TN = []
    FN = []
    for i in range(len(label_test)):
        if label_test[i]==1 and predict[i]==label_test[i]:
                TP.append(i)
        
        elif label_test[i]==1 and predict[i]!=label_test[i]:
                FN.append(i)
        
        elif label_test[i]==0 and predict[i]==label_test[i]:
                TN.append(i)
        
        elif label_test[i]==0 and predict[i]!=label_test[i]:
                FP.append(i)
                
    print(len(TP))
    print(len(FP))
    print(len(TN))
    print(len(FN))

# Scatter plot of the models
def Scatter_Plot():
    fig = plt.figure(figsize=(15,15))

    xs = select_time['3Dcen-1'].to_numpy()
    ys = select_time['3Dcen-2'].to_numpy()

    xs2 = select_time2['3Dcen-1'].to_numpy()
    ys2 = select_time2['3Dcen-2'].to_numpy()

    xTP = []
    yTP = []
    for j in TP:
        x = xyz_test[j][0]
        xTP.append(x)
        y = xyz_test[j][1]
        yTP.append(y)

    xTN = []
    yTN = []
    for j in TN:
        x = xyz_test[j][0]
        xTN.append(x)
        y = xyz_test[j][1]
        yTN.append(y)

    xFP = []
    yFP = []
    for j in FP:
        x = xyz_test[j][0]
        xFP.append(x)
        y = xyz_test[j][1]
        yFP.append(y)

    xFN = []
    yFN = []
    for j in FN:
        x = xyz_test[j][0]
        xFN.append(x)
        y = xyz_test[j][1]
        yFN.append(y)
          
    plt.plot(xTN,yTN,'y.')
    plt.plot(xTP,yTP,'b.')
    plt.plot(xFP,yFP,'g.')
    plt.plot(xFN,yFN,'k.')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('stair detection')
    plt.grid(True)
    plt.draw()

# Plot the Confusion Matrix for the model
def Confusion_Matrix():
    cm_data = [[len(TN), len(FP)], [len(FN), len(TP)]]
    sns.heatmap(cm_data, annot=True, cmap='Blues', fmt='d')

########################################################################
########################### MACHINE LEARNING ###########################
########################################################################

## LINEAR REGRESSION ##
def linear_regression():
    # Preprocessing the data
    xyz_n = preprocessing.scale(xyz_total)
    print('normalized xyz:\n',xyz_n,'\n')
    print('old xyz:\n',xyz_total)
    xyz_train, xyz_test, label_train, label_test = train_test_split(xyz_total, label_total, random_state=4)
    model = LinearRegression()
    # Fit the data
    model.fit(xyz_train, label_train) 
    # Predicted value
    print(model.predict(xyz_test[:5,:]))
    # Actual value
    print(label_total[:5])
    # R^2 coefficient of determination-->scoring to rate how good the model is
    print('the accuracy of linear regression model:',model.score(xyz_total, label_total)*100,'%') 
    model = LinearRegression()
    model.fit(xyz_total, label_total)
    print('the accuracy of linear regression model:',model.score(xyz_n, label_total)*100,'%') 
    predict = model.predict(xyz_test)
    # To see the true positive, false positive, true negative and false negative
    Sensitivity_Test()
    # Plot confusion matrix for linear regression model
    Confusion_Matrix()
    # Plot the predicted linear regression model on the hallway
    Scatter_Plot()
    # Get f1 score
    f1 = f1_score(label_test, model.predict(xyz_test), average = 'macro')
    print('the f1 score of the linear regression model:',f1*100,'%')
    # Test the ROC score
    label_predict = model.predict(xyz_test)
    roc_score = roc_auc_score(label_test, label_predict)
    print('the roc score = ', roc_score*100,'%')
    print('the linear regression score = ',model.score(xyz_test, label_test)*100,'%')

## KNN ##
def knn():
     # test train split
    xyz_train, xyz_test, label_train, label_test = train_test_split(xyz_total, label_total, random_state=4)
    model = KNeighborsClassifier(n_neighbors = 5) #consider5 near points and get y
    # Fit the data
    model.fit(xyz_train, label_train)
    predict = model.predict(xyz_test)
    print(predict)
    print(label_test)
    # To see the true positive, false positive, true negative and false negative
    Sensitivity_Test()
    # Plot the predicted KNN model on the hallway
    Scatter_Plot()
    # Plot confusion matrix for KNN model
    Confusion_Matrix()
    # Get f1 score
    f1 = f1_score(label_test, model.predict(xyz_test), average = 'macro')
    print('the f1 score of the KNN model:',f1*100,'%')
    # Test the ROC score
    label_predict = model.predict(xyz_test)
    roc_score = roc_auc_score(label_test, label_predict)
    print('the roc score = ', roc_score*100,'%')
    print('the knn score = ',model.score(xyz_test, label_test)*100,'%')

## NAIVE BAYES ##
def naive_bayes():
    xyz_train, xyz_test, label_train, label_test = train_test_split(xyz_total,label_total,random_state=4)
    model = GaussianNB()
    # fit the model with the training data
    model.fit(xyz_train,label_train)
    # predict the target on the train dataset
    predict_train = model.predict(xyz_train)
    print('Target on train data',predict_train) 
    # Accuray Score on train dataset
    accuracy_train = accuracy_score(label_train, predict_train)
    print('accuracy_score on train dataset : ', accuracy_train)
    # predict the target on the test dataset
    predict_test = model.predict(xyz_test)
    print('Target on test data',predict_test) 
    # Accuracy Score on test dataset
    accuracy_test = accuracy_score(label_test, predict_test)
    print('accuracy_score on test dataset : ', accuracy_test)
    # To see the true positive, false positive, true negative and false negative
    Sensitivity_Test()
    # Plot the predicted Naive Bayes model on the hallway
    Scatter_Plot()
    # Plot confusion matrix for NAIVE BAYES model
    Confusion_Matrix()
    # Get f1 score
    f1 = f1_score(label_test, model.predict(xyz_test), average = 'macro')
    print('the f1 score of the naive bayes classifier model:',f1*100,'%')
    # Test the ROC score
    label_predict = model.predict(xyz_test)
    roc_score = roc_auc_score(label_test, label_predict)
    print('the roc score = ', roc_score*100,'%')
    print('the linear regression score = ',model.score(xyz_test, label_test)*100,'%')



