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

np.random.seed(123)  # for reproducibility

# Loading training data

features = np.loadtxt('Features.txt')

with open('Labels.txt', 'r') as infile2:
    labels = [line.strip() for line in infile2]

#Loading test data

testfeatures = np.loadtxt('TestFeatures.txt')

with open('TestLabels.txt', 'r') as infile2:
    testlabels = [line.strip() for line in infile2]

# Data preprocessing

# 1. Encoding
onehotencoder = OneHotEncoder(categorical_features=[1])
X_train = onehotencoder.fit_transform(features).toarray()
X_test = onehotencoder.fit_transform(testfeatures).toarray()
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(labels)
y_test = labelencoder.fit_transform(testlabels)

# 2. Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# 3. Vectors to matrices
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]

# 4. Matrices to tensors
X_train = X_train.reshape(-1, 1,314,1)
X_test = X_test.reshape(-1,1,314,1)

# 5. Categorizing labels
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Parameters
batch_size = 56
epochs = 20
num_classes = 14

# Network setup
model = Sequential()

# Layer composition
model.add(Conv2D(28, kernel_size=(1,2),
                 activation='linear',
                 input_shape=(1,314,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(56, (1,2), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(112, (1,2), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(output_dim=128, return_sequences=True))
def td_avg(x):
    return th.mean(x, axis=1)
def td_avg_shape(x):
    return tuple((batch_size,14))
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
model.add(Lambda(td_avg,output_shape=td_avg_shape))


# Compiling
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model summary
print(model.summary())

#Training the model
model_train = model.fit(X_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, verbose=1)

#Saving the model
model.save('Model.h5') # Model
model.save_weights('Model_weights.h5') #Weights
model_json = model.to_json() #Architecture
with open("Model_architecture.json", "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")

# Evaluating the model on test data
scores = model.evaluate(X_test, y_test_one_hot, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Results on separate classes (see README.txt)
predicted_classes = model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
