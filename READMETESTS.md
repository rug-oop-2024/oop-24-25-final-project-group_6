# How to use the program

This file explains how to use the streamlit program for testing. The 1 dataset you can use is contained in the datasets folder. Furthermore, there are prediction csv files in the datasets folder, which show what target value they are predicting and what dataset they use. Assume for these prediction dataset that all input features are selected except the target feature. To begin with the process go into the datasets subpage.

# Datasets page
First of all select the dataset you want to use by pressing browse files, then go into the datasets dictonary and choose the csv file of your choice. Two buttons show up. The first button is for downloading the dataset, the second button "save" saves the artifact dataset and the dataset object in the assets folder so you can later use them in the modelling process. After saving the dataset you can delete the dataset using the "delete dataset" button if you do not want to use the dataset anymore. Proceed to the modelling page if you have saved all the datasets you want to use for testing the program.

# Modelling page
In here you can use the previously selected dataset to train and test a model. First of all select the dataset you want to use, then select the target column for which the model gets trained. Furthermore, select input columns which are used to predict the target column. Then, select the model you want to use for predicting the target column. Select the training-test split you want to use. Then the metrics, it is recommended to at least use the r_squared for regression models and accuracy for classifaction model as these metrics most accurately describe the validness of the model. 

When you have selected all these options, a pipeline summary will be showed. Here you can see in a more aesthetic way which parameters are selected or which parameters still needs to be selected.

If you think all parameters are just right, you can procceed to train the pipeline with the "Train pipeline" button. If the training proccess concluded, the pipeline results will be shown, which give insights on the metrics whether the pipeline accurately predicts the target feature. If the metric values are too low, under the pipeline results an error will be raised, which shows which metric is too low. You cannot procceed with such error, as the prediction of target values will not be accurate enough under a certain threshold.

Finally, you can save the pipeline by giving it a name and a version. A version needs to be formatted which 3 digits each seperated by a dot, thus 1.0.0; Then press save pipeline if you think you have an appropriate name and version. Proceed to the deployment page if you have saved all the pipelines you want to use for tesing the program.

# Deployment page
Select the pipeline you want to use. Then scroll down to the predictions section. Where you can select a csv file for predicting values. Make sure all input features with the right feature column names are contained in the file. If you used my datasets, make sure you had selected every input feature in the previous modelling section, as they are only compatible with all input features selected. Then a "predict" button show up, which is for starting the process of predicting. Press the button and see the results!

Congratulations, you now know and have properly tested the program!