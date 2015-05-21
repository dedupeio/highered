import pyhacrf
from pyhacrf.pyhacrf import Hacrf, StringPairFeatureExtractor

import numpy


class CRFEditDistance(object) :
    def __init__(self) :
        self.model = Hacrf(l2_regularization=1.0)
        self.model.parameters = numpy.array(
            [[-0.06307111, -0.35686125, -0.24368323],
             [ 0.05599825,  0.02427958,  0.24368323],
             [-0.08365461, -0.01643165, -0.02468913],
             [ 0.04300183, -0.03985499,  0.03919838],
             [ 0.08278298,  0.16122849, -0.06341355],
             [-0.04207757, -0.27139228,  0.04556619],
             [ 0.17641326, -0.25142443, -0.15558055],
             [-0.13576664,  0.08529407,  0.15891866]])
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
        if not string_1 or not string_2 :
            return numpy.nan
        features = self.feature_extractor.fit_transform(((string_1, string_2),))
        return self.model.predict_proba(features)[0,1]
