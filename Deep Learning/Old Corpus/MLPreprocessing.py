import numpy as np
import keras
import tensorflow as tf
from keras import layers
from keras import backend as th
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report

#1. Loading data. Both labels and features can be obtained via 'vecsnfeatures' function from DocumentProcessingPipeline.py

features = np.loadtxt('Features.txt')

with open('Labels.txt', 'r') as infile2:
     labels = [line.strip() for line in infile2]

print(features.shape)
#2. Encoding

onehotencoder = OneHotEncoder(categorical_features=[1])
X_train = onehotencoder.fit_transform(features).toarray()
X_test = onehotencoder.fit_transform(testfeatures).toarray()
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(labels)
y_test = labelencoder.fit_transform(testlabels)

#3. Feature scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#4. Vectors to matrixes
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]

#5. Features to tensors
X_train = X_train.reshape(-1, 1,314,1)
X_test = X_test.reshape(-1,1,314,1)

#6. Labels to categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)