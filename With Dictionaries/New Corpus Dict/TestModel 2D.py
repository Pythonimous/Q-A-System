import numpy as np
import keras


from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pickle
from keras.utils.vis_utils import plot_model


# Loading test data


with open('TestMatrices.pkl', 'rb') as testfile:
    testfeatures = pickle.load(testfile)

with open('TestLabels.txt', 'r') as infile2:
    testlabels = [line.strip() for line in infile2]


labelencoder = LabelEncoder()
y_test = labelencoder.fit_transform(testlabels)
y_test_one_hot = to_categorical(y_test)

X_test = np.array(testfeatures)

num_classes = 13


def test_model(model):
    model = keras.models.load_model(model)

    # Model summary
    print(model.summary())

    plot_model(model, to_file='structure.png')

    # Evaluating the model on a test data
    scores = model.evaluate(X_test, y_test_one_hot, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Results on separate classes (see README.txt)
    predicted_classes = model.predict(X_test)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    target_names = ["Class {}".format(i) for i in range(num_classes)]

    print(classification_report(y_test, predicted_classes, target_names=target_names))