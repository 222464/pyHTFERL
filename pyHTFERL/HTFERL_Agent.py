import HTFERL_Hierarchy as hth
import numpy as np
from enum import Enum

class InputType(Enum):
    state = 0
    action = 1
    unused = 2

class HTFERL_Agent(object):
    """HTFERL Agent"""

    def __init__(self, layerParams, inputTypes, minInitWeight, maxInitWeight):
        self.hierarchy = hth.HTFERL_Hierarchy(layerParams, minInitWeight, maxInitWeight)

        self.prevValue = 0.0

        self.qLayers = []

        for l in range(0, len(self.hierarchy.layers)):
            qLayer = np.zeros(self.hierarchy.layers[l].shape)

            self.qLayers.append(qLayer)

        self.actionIndices = []
        self.unusedIndices = []

        assert(len(inputTypes) == len(self.hierarchy.layers[0].visibleStates))

        for i in range(0, len(inputTypes)):
            if inputTypes[i] == InputType.action:
                self.actionIndices.append(i)
            elif inputTypes[i] == InputType.unused:
                self.unusedIndices.append(i)

    def getQ(self):
        sum = 0.0

        for l in range(0, len(self.hierarchy.layers)):
            sum += np.sum(self.hierarchy.layers[l].hiddenFeedForwardStates * self.qLayers[l])

        return sum

    def updatePrevQ(self, error):
        for l in range(0, len(self.hierarchy.layers)):
            self.qLayers[l] += error * self.hierarchy.layers[l].hiddenFeedForwardStatesPrev

    def setInput(self, position, state):
        self.hierarchy.setInput(position, state)
    
    def getAction(self, position):
        return self.hierarchy.getReconstruction(position)

    def step(self, reward, hAlpha, hBeta, qAlpha, gamma, actionBreakChance):
        self.hierarchy.activateFeedForward()

        q = self.getQ()

        tdError = reward + gamma * q - self.prevValue

        self.prevValue = q

        self.updatePrevQ(tdError * qAlpha)

        if tdError > 0.0:
            self.hierarchy.learn(hAlpha, hBeta)

        self.hierarchy.stepEnd()

        self.hierarchy.activateFeedBack()

        # Exploration
        for i in range(0, len(self.actionIndices)):
            if np.random.rand() < actionBreakChance:
                self.hierarchy.layers[0].visibleReconstruction[i] = np.random.rand()

