Deep Learning folder:

///

Old corpus folder: - with unequal class distribution

***

New corpus folder: less classes, semi-normalized distribution, sentences are cut after average size

Classification.docx: classes' clarification;

Training.txt, Test.txt - corpora

TestFeatures.pkl, TestLabels.txt, TrainingFeatures.pkl, TrainingLabels.txt  - preprocessed for 2D ML training files (reshaping in TrainModel.py, TestModel.py)

Preprocess.py - preprocessing algorithm
TrainModel.py - training model
TestModel.py - loading and testing model separately

Top Model - contains best model and .txt file, explaining the name
CLASSIFICATION.txt - matches inside model (Python) classes and their respective real classes

---

With Dictionaries folder:

***

Dictionary Preprocessing: folder with differently extracted features and feature extraction algorithm: parts exceeding the average size are numpy.mean'ed; dictionaries are used.

To Word Vectors.py: represents sentences as lists of POS tags (creates ~Raw.pkl -> ~POS.pkl for training / test for any given file) + creates FeaturesDictionary.pkl: dictionary with pairs: POS-tagged word - embedding
To Vectors.py, To Matrices.py, To Tensors.py: use created POS.pkl and Dictionary.pkl to make representations for 1D, 2D and 3D CNN respectively
Training.txt, Test.txt - training and test corpora

***

New Corpus Dict folder: different preprocessing mechanism (clarified above) is used.

TestMatrices.pkl, TestLabels.txt, TrainingMatrices.pkl, TrainingLabels.txt  - preprocessed for 2D ML training files

TrainModel 2D.py - training model
TestModel 2D.py - loading and testing model separately
Calibrate 2D.py - tuning parameters (GridSearchCV)

CLASSIFICATION.txt - similar to in New Corpus
