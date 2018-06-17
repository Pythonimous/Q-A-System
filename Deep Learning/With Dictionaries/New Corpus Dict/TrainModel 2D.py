import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import pickle

np.random.seed(123)  # for reproducibility

# Loading training data

with open('TrainingMatrices.pkl', 'rb') as trainfile:
    features = pickle.load(trainfile)

with open('TrainingLabels.txt', 'r') as infile2:
    labels = [line.strip() for line in infile2]

# Loading test data

with open('TestMatrices.pkl', 'rb') as testfile:
    testfeatures = pickle.load(testfile)

with open('TestLabels.txt', 'r') as infile2:
    testlabels = [line.strip() for line in infile2]

# Data preprocessing


labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(labels)
y_test = labelencoder.fit_transform(testlabels)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


X_train = np.array(features)
X_test = np.array(testfeatures)


# Parameters
batch_size = 512
epochs = 100
num_classes = 13
window = (20,3)

# Network setup
model = Sequential()

# Layer composition
model.add(Conv2D(26, kernel_size=window,
                 activation='linear',
                 input_shape=(340,8,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))


# Compiling
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Model summary
print(model.summary())

earlystop = EarlyStopping(monitor='acc', min_delta = 0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop] #If accuracy has not changed for 5 epochs.

#Training the model
model_train = model.fit(X_train, y_train_one_hot, batch_size=batch_size, callbacks = callbacks_list, epochs=epochs, verbose=1)

#Saving the model
model.save('26CNN512B100EwDOwTD 20,3 - Early Stop - DO 0.2.h5') # Model

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
