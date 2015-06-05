import pyhacrf
from pyhacrf import Hacrf, StringPairFeatureExtractor

import numpy


class CRFEditDistance(object) :
    def __init__(self) :
        self.model = Hacrf(l2_regularization=1.0)
        self.model.parameters = numpy.array(
            [[ 0.9416244,   0.08417879,  0.,        ],
             [ 0.15201385,  1.63699504,  0.,        ],
             [ 0.02203581,  0.01057592,  0.,        ],
             [-0.17404862,  0.26152985,  0.,        ],
             [-0.29091906,  0.45691683,  0.,        ],
             [ 0.44293187,  0.06393411,  0.,        ],
             [ 0.08997326,  0.07139777,  0.,        ],
             [ 0.06203954,  0.85681935,  0.,        ]])
        self.model.classes = ['match', 'non-match']

        self.model._state_machine =\
            pyhacrf.pyhacrf.DefaultStateMachine(self.model.classes)

        self.feature_extractor = StringPairFeatureExtractor(match=True,
                                                            numeric=True)

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
        if not string_1 or not string_2 :
            return numpy.nan
        if len(string_1) > len(string_2) :
            string_1, string_2 = string_2, string_1
        features = self.feature_extractor.fit_transform(((string_1, string_2),))
        return self.model.predict_proba(features)[0,1]
