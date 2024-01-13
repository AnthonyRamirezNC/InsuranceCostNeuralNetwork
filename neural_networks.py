import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np

tensorflow.random.set_seed(35) #for the reproducibility of results

def design_model(trainX, trainY):
  model = DecisionTreeRegressor()
  # Fit the model to the training data using 500 epochs, and 1 batch size
  model.fit(trainX, trainY)
  return model

dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split the data into training and test data
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

# Convert data types
features_train = features_train.astype('float32')
labels_train = labels_train.astype('float32')
features_test = features_test.astype('float32')
labels_test = labels_test.astype('float32')

# Check for NaN and Inf values
if np.isnan(features_train).any() or np.isnan(labels_train).any() or np.isinf(features_train).any() or np.isinf(labels_train).any():
    raise ValueError("Data contains NaN or Inf values.")

# Invoke the function for our model design
model = design_model(features_train, labels_train)
# #evaluate the model on the test data
# val_mse, val_mae = model.evaluate(features_test, labels_test, verbose = 0)

# print("MAE: ", val_mae)

# #print model prediction function
# print(model.pred)


# make a prediction with the model using example data
# print('predicting cost with the following data:')
# print("age: 61")
# print("sex: 1")
# print("bmi: 29.07")
# print("children: 0")
# print("smoker: 1")
# print("region: northwest")
# X = pd.read_csv("test.csv")
# X = ct.transform(X)
print(model.predict(features_test))

print("Answers: \n" + str(labels_test))

