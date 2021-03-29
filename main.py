# Importing libraries for diabetes classification
import pandas as pd  # to read a dataset

from sklearn.neural_network import MLPClassifier  # import classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Read a dataset from National Institute of Diabetes and Digestive Health
df = pd.read_csv("diabetes.csv")
print(df)
print(df.head())  # to visualise first few rows from datasets
df.info()  # to get some info about that data
df.corr().T  # to find the correlation between the features
x = df[["Glucose"]]    # Assign the x - co ordinate with Glucose
y = df[["Outcome"]]    # Assign Y - co ordinate with the outcome
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3) # train and test the dataset
print(x_train.shape)
print(y_train.shape)
print(df.corr().T)
# initialise the classifier
clf = MLPClassifier(hidden_layer_sizes=(51),solver = "lbfgs",alpha =1e-5,activation ="logistic")
clf.fit(x,y)  # fit the data
y_predict =clf.predict(x_train)  # predict the data
print('Accuracy Score : ')
print ( accuracy_score(y_train, y_predict)) # calculate the accuracy score , classification report , confusion matrix
print('Classification Report : ')

print ( classification_report(y_train, y_predict))
print('Confusion Matrix : ')

print( confusion_matrix(y_train,y_predict))
print('\n')
n = int(input("Enter the blood glucose level :  "))  # ge the input blood glucose level
a =clf.predict(([[n]]))
if (n < 90):
     print (" You have a low blood pressure")
elif(a==0):
    print("You are diabetic free")

else :
    print("You have diabetes , consult a doctor")

1