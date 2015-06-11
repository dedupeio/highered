import pyhacrf
from pyhacrf import Hacrf, StringPairFeatureExtractor

import numpy


class CRFEditDistance(object) :
    def __init__(self) :
        self.model = Hacrf(l2_regularization=1.0)
        self.model.parameters = numpy.array(
            [[ 0.95778803,  1.12747085,  0.0107147 ],
             [-0.03295845,  2.14652662, -0.21984459],
             [ 0.28332966, -0.03945445,  0.14360488],
             [ 0.06154703,  1.1877714,   0.70521627],
             [-0.68625725,  0.95054487, -0.3146599 ],
             [ 0.34178646, -0.29681464, -0.42189039],
             [-0.49620439,  0.63679766,  0.21795662],
             [ 0.15113431,  0.8799614,  -0.52835432]])
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
