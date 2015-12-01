import pyhacrf
from pyhacrf import Hacrf, StringPairFeatureExtractor
from pyhacrf.state_machine import DefaultStateMachine

import numpy as np

class WiderStateMachine(DefaultStateMachine) :
    BASE_LENGTH = 90

    def _lattice_ends(self) :
        lattice_limits = {}

        lengths = np.arange(self.BASE_LENGTH)
        lengths.reshape(1, -1)

        I = self._base_lattice[..., 3:4] < lengths
        for i in range(self.BASE_LENGTH) :
            lattice_limits[i, None] = I[..., i].nonzero()[0]

        J = self._base_lattice[..., 4:5] < lengths

        IJ = np.expand_dims(I, axis=0).T & J

        for i in range(self.BASE_LENGTH) :
            for j in range(self.BASE_LENGTH) :
                if i <= j :
                    lattice_limits[i,j] = IJ[i, ..., j].nonzero()[0]

        return lattice_limits



class CRFEditDistance(object) :
    def __init__(self) :
        self.model = Hacrf(l2_regularization=1.0)
        self.model.parameters = np.array(
            [[-1.14087105,  2.41450373, -0.42000576],
             [-0.0619002,   0.79430259,  0.33864121],
             [-0.25353303,  1.69376742,  0.71731646],
             [ 0.31544095,  1.47012227, -0.39960507],
             [ 0.51356569, -0.67293917, -0.56861512],
             [-0.57547361,  0.57599782,  0.3115221 ],
             [ 0.55744877,  0.16423292, -0.64028285],
             [-0.61935669, -0.02237494,  0.49829992]])
        self.model.classes = ['match', 'non-match']

        self.model._state_machine = WiderStateMachine(self.model.classes)

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
        if len(string_1) > len(string_2) :
            string_1, string_2 = string_2, string_1
        features = self.feature_extractor.fit_transform(((string_1, string_2),))
        return self.model.predict_proba(features)[0,1]
