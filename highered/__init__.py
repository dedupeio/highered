import pyhacrf
from pyhacrf.pyhacrf import Hacrf
from pyhacrf.features import StringPairFeatureExtractor

import numpy


class CRFEditDistance(object) :
    def __init__(self) :
        self.model = Hacrf(l2_regularization=1.0)
        self.model.parameters = numpy.array([[-0.08833188,  0.  ],
                                             [ 0.16700528,  0.  ],
                                             [-0.11705577,  0.  ],
                                             [ 0.14053809,  0.  ],
                                             [-0.20092105,  0.  ],
                                             [ 0.2766846 ,  0.  ],
                                             [ 0.19652368,  0.  ],
                                             [-0.21709615,  0.  ]])
        self.model.classes = ['match', 'non-match']

        (self.model._state_machine, 
         self.model._states_to_classes) =\
                self.model._default_state_machine(self.model.classes)



        self.feature_extractor = StringPairFeatureExtractor(match=True,
                                                            numeric=True)

    def train(self, examples, labels) :
        extracted_examples = self.feature_extractor.fit_transform(examples)
        self.model.fit(extracted_examples, labels, verbosity=1)

    def __call__(self, string_1, string_2) :
        features = self.feature_extractor.fit_transform(((string_1, string_2),))
        return self.model.predict_proba(features)[0,1]
