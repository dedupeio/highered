import pyhacrf
from pyhacrf import Hacrf, StringPairFeatureExtractor
from pyhacrf.state_machine import DefaultStateMachine
from pyhacrf.adjacent import forward_predict

import numpy as np

class CRFEditDistance(object) :
    def __init__(self) :
        classes = ['match', 'non-match']
        self.model = Hacrf(l2_regularization=100.0,
                           state_machine=DefaultStateMachine(classes))
        self.model.parameters = np.array(
            [[-0.22937526,  0.51326066],
             [ 0.01038001, -0.13348901],
             [-0.03062821,  0.13769178],
             [ 0.02024813, -0.01835538],
             [ 0.09208272,  0.15466022],
             [-0.08170265, -0.02484392],
             [-0.01762858,  0.17504624],
             [ 0.02800866, -0.04442708]],
            order='F')
        self.parameters = self.model.parameters.T
        self.model.classes = ['match', 'non-match']

        self.feature_extractor = StringPairFeatureExtractor(match=True,
                                                            numeric=False)


        
    def fast_pair(self, x):
        x_dot_parameters = np.matmul(x, self.parameters)

        probs = forward_predict(x_dot_parameters, 2)

        return probs


    def train(self, examples, labels) :
        examples = [(string_2, string_1) 
                    if len(string_1) > len(string_2)
                    else (string_1, string_2)
                    for string_1, string_2
                    in examples]
        print(examples)
        extracted_examples = self.feature_extractor.fit_transform(examples)
        self.model.fit(extracted_examples, labels, verbosity=1)

    def __call__(self, string_1, string_2) :
        if len(string_1) > len(string_2) :
            string_1, string_2 = string_2, string_1
        array1 = np.array(tuple(string_1)).reshape(-1, 1)
        array2 = np.array(tuple(string_2)).reshape(1, -1)
        features = self.feature_extractor._extract_features(array1, array2)
        return self.fast_pair(features)[1]
