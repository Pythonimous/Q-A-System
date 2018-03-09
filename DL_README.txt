Document processing pipeline: function pipeline, designed to separate delimited questions and labels, extracting features and vectorally representing the former.
MLPreprocessing: transforming labels and features into format, suitable for CNN
TrainModel: training network configuration and evaluating on existing test data
TestModel: complying network, loading weights instead of re-training, with testing possibilities.
It is also possible to load trained model using standard methods (loading full model / loading architecture and assigning weights); however, due to the network specifics (namely it having custom Lambda layer added using methods of backend library: TensorFlow), current version of Keras is incapable of processing it correctly.
Evaluation is possible if the architecture is compiled without training and loading existing weights of pre-trained model.
The model itself does not suffice to volume restrictions by softconf submit system, and thus uploading weights along with the source code is much more info-efficient.
During evaluation visualization, the model names classes in its own system. Here are model's classification analogues:
Class0 = 1
Class1 = 10
Class2 = 11
Class3 = 12
Class4 = 13
Class5 = 14
Class6 = 2
Class7 = 3
Class8 = 4
Class9 = 5
Class10 = 6
Class11 = 7
Class12 = 8
Class13 = 9