Deep Learning folder:

Old corpus - with unequal class distribution
New corpus - less classes, semi-normalized distribution

Classification.docx: classes' clarification;
Training.txt, Test.txt - training and test corpora
TestFeatures.pkl, TestLabels.txt, TrainingFeatures.pkl, TrainingLabels.txt  - semi-preprocessed for ML-training: lemmatization and POS-tagging (in preprocessing pipeline) take a long time on their own; reshaping - in TrainModel.py file
Preprocess.py - pipeline for processing labeled text (with delimiters; corpus delimiter is \t)
TrainModel.py - training model
TestModel.py - loading and testing model separately
Top Model - contains best model, weights, architecture and .txt file, explaining the name
CLASSIFICATION.txt - matches inside model (Python) classes and their respective real classes