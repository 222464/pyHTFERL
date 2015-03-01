import HTFERL_Layer as htl
import numpy as np

class HTFERL_LayerParams(object):
    """Container for parameters for initializing HTFERL layers"""

    inputSize = (16, 16)
    layerSize = (16, 16)
    feedForwardRadius = 2
    lateralRadius = 2
    feedBackRadius = 2
    inhibitionRadius = 2
    sparsity = 0.125

class HTFERL_Hierarchy(object):
    """A Hierachy of Layers for HTFERL"""
    
    def __init__(self, layerParams, minInitWeight, maxInitWeight):
        self.layers = []

        for l in range(0, len(layerParams)):
            layer = htl.HTFERL_Layer(layerParams[l].inputSize,
                                     layerParams[l].layerSize,
                                     layerParams[l].feedForwardRadius,
                                     layerParams[l].lateralRadius,
                                     layerParams[l].feedBackRadius,
                                     layerParams[l].inhibitionRadius,
                                     layerParams[l].sparsity,
                                     minInitWeight, maxInitWeight)

            # Make sure input of one layer is the size of the hidden layer below it
            assert(l == 0 or len(layer.visibleStates) == len(self.layers[len(self.layers) - 1].hiddenFeedForwardStates))

            self.layers.append(layer)

    def setInput(self, position, state):
        self.layers[0].setVisibleState(position, state)

    def getReconstruction(self, position):
        return self.layers[0].getVisibleReconstruction(position)

    def activateFeedForward(self):
        # Up pass
        for l in range(0, len(self.layers)):
            if l != 0:
                for i in range(0, len(self.layers[l].visibleStates)):
                    self.layers[l].visibleStates[i] = self.layers[l - 1].hiddenFeedForwardStates[i]

            self.layers[l].activateForward()

    def activateFeedBack(self):
        # Down pass
        for lb in range(0, len(self.layers)):
            l = len(self.layers) - lb - 1

            if lb == 0:
                self.layers[l].activateFeedBack(np.zeros(self.layers[l].hiddenFeedBackStates.shape))
            else:
                self.layers[l].activateFeedBack(self.layers[l + 1].hiddenFeedBackStates)
        
    def learn(self, alpha):
        for l in range(0, len(self.layers)):
            if l == len(self.layers) - 1:
                self.layers[l].learn(self.layers[l].visibleStates, np.zeros(self.layers[l].hiddenFeedBackStates.shape), alpha)
            else:
                self.layers[l].learn(self.layers[l].visibleStates, self.layers[l + 1].hiddenFeedBackStates, alpha)

    def stepEnd(self):
        for l in range(0, len(self.layers)):
            self.layers[l].stepEnd()