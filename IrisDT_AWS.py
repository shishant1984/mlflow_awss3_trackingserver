#Iris dataset with Decision Tree classifier for new exeriment in MLflowimport mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://ec2-65-0-75-164.ap-south-1.compute.amazonaws.com:5000/') # Setting up mlflow tracking server locally
#Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


#Split data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.2 ,random_state= 42)

#Parameters for random forest

max_depth = 10


#apply mlflow

mlflow.set_experiment('IrisDecisionTree') # naming experiment

with mlflow.start_run(): # Its to tell what all to log by mlflow
    irisdt = DecisionTreeClassifier(max_depth= max_depth)
    irisdt.fit (X_train,y_train)
    y_pred = irisdt.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    # This is logging code 
    mlflow.log_metric('accuracy',accuracy) #log metric
    mlflow.log_param('max_depth',max_depth) # log parameters
    # You can track code , model , artifacts as well
   
   #Create coonfusion matrix
    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues' ,xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matric')

    plt.savefig('Confusion_matrix.png') # Save plot as artifact
    mlflow.log_artifact("Confusion_matrix.png") # log artifacts with path to file
    mlflow.log_artifact(__file__) # log this code file as well
    mlflow.sklearn.log_model(irisdt,'Iris Decision Tree') # log model .
    mlflow.set_tag('CreatedBY','Shishant') # Tags for searching when lot of runs in place
    mlflow.set_tag('AlgoUsed','DecisionTree')

    print(accuracy)    
