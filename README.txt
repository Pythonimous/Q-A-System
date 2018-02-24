Training.txt, Test.txt - training and test corpora
TestLabels.txt, TestFeatures.txt - test corpora for ML-preprocessing (it requires separate files)
Features.txt, Labels.txt - already preprocessed for ML-training: lemmatization and POS-tagging (in preprocessing pipeline) take a long time on their own.
DocumentProcessingPipeline.py - pipeline for processing labeled text (with separators; default used separator is \t, modifiable)
MLPreprocessing.py - converting preprocessed text into ML .fit format (is included in both test and training algorithms)
TrainModel.py - training model
TestModel.py - loading and testing model separately
Model_Weights.h5 - check DL_README.txt for details
SVM_Linear files - last year's baseline algorithm created with different set of features using RapidMiner.
Classification.docx - explanation on every question type.