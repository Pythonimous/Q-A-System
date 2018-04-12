Deep Learning folder:

Old corpus - with unequal class distribution
New corpus - less classes, semi-normalized distribution

Training.txt, Test.txt - training and test corpora
TestLabels.txt, TestFeatures.txt - test corpora for ML-preprocessing (it requires separate files)
Features.txt, Labels.txt - already preprocessed for ML-training: lemmatization and POS-tagging (in preprocessing pipeline) take a long time on their own.
DocumentProcessingPipeline.py - pipeline for processing labeled text (with separators; default used separator is \t, modifiable)
DocumentProcessingPipeline tf-idf.py - pipeline with tf-idf usage
MLPreprocessing.py - converting preprocessed text into ML .fit format (is included in both test and training algorithms)
TrainModel.py - training model
TestModel.py - loading and testing model separately
Model_Weights.h5 - check DL_README.txt for details