import numpy as np
import keras

from keras import backend as th
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle


with open('TestFeatures.pkl', 'rb') as testfile:
    testfeatures = pickle.load(testfile)

with open('TestLabels.txt', 'r') as infile2:
    testlabels = [line.strip() for line in infile2]

# Data preprocessing


labelencoder = LabelEncoder()
y_test = labelencoder.fit_transform(testlabels)

# 2. Feature scaling
sc = StandardScaler()
X_test = [sc.fit_transform(i) for i in testfeatures] #scaling matrices

X_test = np.array([i.reshape((1,)+i.shape) for i in X_test])

X_test = X_test.reshape(181,340,8,1)

y_test_one_hot = to_categorical(y_test)

num_classes = 13

model = keras.models.load_model('26CNN512B100EwDOwTD 20,3 - Early Stop 67 - DO 0.2.h5')

# Model summary
print(model.summary())

# Evaluating the model on a test data
scores = model.evaluate(X_test, y_test_one_hot, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Results on separate classes (see README.txt)
predicted_classes = model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))