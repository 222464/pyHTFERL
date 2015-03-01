import numpy as np

def getSubarray(matrix, lowerBound, upperBound):
    size = (upperBound[0] - lowerBound[0]) * (upperBound[1] - lowerBound[1])

    result = np.zeros(size)

    index = 0

    for x in range(lowerBound[0], upperBound[0]):
        for y in range(lowerBound[1], upperBound[1]):
            if x >= 0 and y >= 0 and x < matrix.shape[0] and y < matrix.shape[1]:
                result[index] = matrix[x][y]
                index += 1

    return result

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class HTFERL_Layer(object):
    """A layer in a HTFERL hierarchy"""

    def __init__(self,
                 inputSize = (16, 16),
                 layerSize = (16, 16),
                 feedForwardRadius = 2, lateralRadius = 3, feedBackRadius = 3, inhibitionRadius = 3, 
                 sparsity = 0.125,
                 minInitWeight = -0.1, maxInitWeight = 0.1):

        self.inputSize = inputSize
        self.layerSize = layerSize

        self.feedForwardRadius = feedForwardRadius
        self.lateralRadius = lateralRadius
        self.inhibitionRadius = inhibitionRadius
        self.feedBackRadius = feedBackRadius

        self.sparsity = sparsity

        self.visibleStates = np.zeros(inputSize)
        self.visibleBiases = np.random.rand(inputSize[0], inputSize[1]) * (maxInitWeight - minInitWeight) + minInitWeight
        self.visibleReconstruction = np.zeros(inputSize)

        self.hiddenSums = np.zeros(layerSize)
        self.hiddenActivations = np.zeros(layerSize)
        self.hiddenFeedForwardStates = np.zeros(layerSize)
        self.hiddenFeedBackStates = np.zeros(layerSize)
        self.hiddenBiases = np.random.rand(layerSize[0], layerSize[1]) * (maxInitWeight - minInitWeight) + minInitWeight
        self.hiddenDutyCycles = np.full(layerSize, sparsity)
        self.hiddenErrors = np.zeros(layerSize)
    
        inputCount = inputSize[0] * inputSize[1]
        hiddenCount = layerSize[0] * layerSize[1]

        feedForwardSize = np.power(2 * self.feedForwardRadius + 1, 2)
        lateralSize = np.power(2 * self.lateralRadius + 1, 2)
        feedBackSize = np.power(2 * self.feedBackRadius + 1, 2)

        self.feedForwardWeights = np.random.rand(hiddenCount, feedForwardSize) * (maxInitWeight - minInitWeight) + minInitWeight
        self.lateralWeights = np.random.rand(hiddenCount, lateralSize) * (maxInitWeight - minInitWeight) + minInitWeight
        self.feedBackWeights = np.random.rand(hiddenCount, feedBackSize) * (maxInitWeight - minInitWeight) + minInitWeight

    def setVisibleState(self, position, state):
        self.visibleStates[position] = state

    def getHiddenState(self, position):
        return self.hiddenFeedForwardStates[position]

    def activateForward(self):
        for x in range(0, self.layerSize[0]):
            for y in range(0, self.layerSize[1]):
                index = x + y * self.layerSize[0]

                # Center of visible receptive field
                vCenter = (int(x / (self.layerSize[0] - 1) * (self.inputSize[0] - 1)), int(y / (self.layerSize[1] - 1) * (self.inputSize[1] - 1)))

                subFeedForward = getSubarray(self.visibleStates,
                                             (vCenter[0] - self.feedForwardRadius, vCenter[1] - self.feedForwardRadius),
                                             (vCenter[0] + self.feedForwardRadius + 1, vCenter[1] + self.feedForwardRadius + 1))

                # TODO: Decide whether to move this to feedback phase instead?
                subFeedLateral = getSubarray(self.hiddenFeedForwardStates,
                                             (x - self.lateralRadius, y - self.lateralRadius),
                                             (x + self.lateralRadius + 1, y + self.lateralRadius + 1))

                self.hiddenSums[x][y] = np.dot(self.feedForwardWeights[index], subFeedForward) + np.dot(self.lateralWeights[index], subFeedLateral) + self.hiddenBiases[x][y]

                self.hiddenActivations[x][y] = sigmoid(self.hiddenSums[x][y])


        # Inhibition
        localActivity = np.round(self.sparsity * np.power(2 * self.feedForwardRadius + 1, 2))

        for x in range(0, self.layerSize[0]):
            for y in range(0, self.layerSize[1]):
                numHigher = 0.0

                for dx in range(-self.inhibitionRadius, self.inhibitionRadius + 1):
                    for dy in range(-self.inhibitionRadius, self.inhibitionRadius + 1):
                        inhibitPosition = (x + dx, y + dy)

                        if inhibitPosition[0] >= 0 and inhibitPosition[1] >= 0 and inhibitPosition[0] < self.layerSize[0] and inhibitPosition[1] < self.layerSize[1]:
                            if self.hiddenSums[inhibitPosition] > self.hiddenSums[x][y]:
                                numHigher += 1.0

                if numHigher < localActivity:
                    self.hiddenFeedForwardStates[x][y] = 1.0
                else:
                    self.hiddenFeedForwardStates[x][y] = 0.0

    def activateFeedBack(self, nextLayerHidden):
        # Get feedback from higher layer and modify SDR with this information
        for x in range(0, self.layerSize[0]):
            for y in range(0, self.layerSize[1]):
                index = x + y * self.layerSize[0]

                # Center of next layer field
                nCenter = (int(x / (self.layerSize[0] - 1) * (nextLayerHidden.shape[0] - 1)), int(y / (self.layerSize[1] - 1) * (nextLayerHidden.shape[1] - 1)))

                subFeedBack = getSubarray(nextLayer,
                                          (nCenter[0] - self.lateralRadius, nCenter[1] - self.lateralRadius),
                                          (nCenter[0] + self.lateralRadius + 1, nCenter[1] + self.lateralRadius + 1))

                self.hiddenSums[x][y] += np.dot(self.feedBackWeights[index], subFeedBack)

                # Recompute
                self.hiddenActivations[x][y] = sigmoid(self.hiddenSums[x][y])

        # Inhibition
        localActivity = np.round(self.sparsity * np.power(2 * self.feedForwardRadius + 1, 2))

        for x in range(0, self.layerSize[0]):
            for y in range(0, self.layerSize[1]):
                numHigher = 0.0

                for dx in range(-self.inhibitionRadius, self.inhibitionRadius + 1):
                    for dy in range(-self.inhibitionRadius, self.inhibitionRadius + 1):
                        inhibitPosition = (x + dx, y + dy)

                        if inhibitPosition[0] >= 0 and inhibitPosition[1] >= 0 and inhibitPosition[0] < self.layerSize[0] and inhibitPosition[1] < self.layerSize[1]:
                            if self.hiddenSums[inhibitPosition] > self.hiddenSums[x][y]:
                                numHigher += 1.0

                if numHigher < localActivity:
                    self.hiddenFeedForwardStates[x][y] = 1.0
                else:
                    self.hiddenFeedForwardStates[x][y] = 0.0

        # Reconstruct visible
        reverseFeedForwardRadii = (np.ceil(self.layerSize[0] / self.inputSize[0] * self.feedForwardRadius), np.ceil(self.layerSize[1] / self.inputSize[1] * self.feedForwardRadius))

        for x in range(0, self.inputSize[0]):
            for y in range(0, self.inputSize[1]):
                index = x + y * self.layerSize[0]

                # Center of hidden receptive field
                hCenter = (int(x / (self.inputSize[0] - 1) * (self.layerSize[0] - 1)), int(y / (self.inputSize[1] - 1) * (self.layerSize[1] - 1)))

                subHidden = getSubarray(self.hiddenFeedForwardStates,
                                        (hCenter[0] - reverseFeedForwardRadii[0], hCenter[1] - reverseFeedForwardRadii[1]),
                                        (hCenter[0] + reverseFeedForwardRadii[0] + 1, hCenter[1] + reverseFeedForwardRadii[1] + 1))

                self.visibleReconstruction[x][y] = np.dot(self.feedForwardWeights[index].T, subHidden) + self.visibleBiases[x][y]

    def weightUpdate(self, nextVisible, alpha):
        reconstructionError = nextVisible - self.visibleReconstruction

        for x in range(0, self.layerSize[0]):
            for y in range(0, self.layerSize[1]):
                index = x + y * self.layerSize[0]

                # Center of visible receptive field
                vCenter = (int(x / (self.layerSize[0] - 1) * (self.inputSize[0] - 1)), int(y / (self.layerSize[1] - 1) * (self.inputSize[1] - 1)))

                # Get hidden error
                subReconstructionError = getSubarray(reconstructionError,
                                                     (vCenter[0] - self.feedForwardRadius, vCenter[1] - self.feedForwardRadius),
                                                     (vCenter[0] + self.feedForwardRadius + 1, vCenter[1] + self.feedForwardRadius + 1))

                self.hiddenErrors[x][y] = np.dot(self.feedForwardWeights[index].T, subReconstructionError)

                # Update feed forward weights
                self.feedForwardWeights[index] += alpha * 0.5 * (reconstructionError * self.hiddenStates[x][y] + self.hiddenErrors[x][y] * subFeedForward)

                # Update hidden biases
                self.hiddenBiases[x][y] += alpha * self.hiddenErrors[x][y]

        # Update recurrent connections (lateral and feedback)
        for x in range(0, self.layerSize[0]):
            for y in range(0, self.layerSize[1]):
                index = x + y * self.layerSize[0]

                # Center of next layer field
                nCenter = (int(x / (self.layerSize[0] - 1) * (nextLayerHidden.shape[0] - 1)), int(y / (self.layerSize[1] - 1) * (nextLayerHidden.shape[1] - 1)))

                self.hiddenErrors[x][y] = np.dot(self.feedForwardWeights[index].T, reconstructionError)

                subFeedLateral = getSubarray(self.hiddenFeedForwardStates,
                                             (x - self.lateralRadius, y - self.lateralRadius),
                                             (x + self.lateralRadius, y + self.lateralRadius))

                subFeedBack = getSubarray(nextLayer,
                                          (nCenter[0] - self.lateralRadius, nCenter[1] - self.lateralRadius),
                                          (nCenter[0] + self.lateralRadius + 1, nCenter[1] + self.lateralRadius + 1))

                # Update lateral weights
                self.feedForwardWeights[index] += alpha * self.hiddenErrors[x][y] * subFeedLateral

                # Update feedback weights
                self.feedBackWeights[index] += alpha * self.hiddenErrors[x][y] * subFeedBack

        # Update visible biases
        self.visibleBiases += alpha * reconstructionError






