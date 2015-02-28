import numpy as np

class HTFERL_Layer(object):
    """A layer in a HTFERL hierarchy"""

    def __init__(self,
                 size = (16, 16),
                 feedForwardRadius = 2, lateralRadius = 3, inhibitionRadius = 3, feedBackRadius = 3,
                 minInitWeight = -0.1, maxInitWeight = 0.1):

        self.feedForwardRadius = feedForwardRadius
        self.lateralRadius = lateralRadius
        self.inhibitionRadius = inhibitionRadius
        self.feedBackRadius = feedBackRadius

        self.visibleStates = np.zeroes(size)
        self.visibleBiases = np.random.rand(size)
        







