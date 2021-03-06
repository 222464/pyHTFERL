import HTFERL_Hierarchy as hth

layerParams = [hth.HTFERL_LayerParams() for i in range(0, 1)]

layerParams[0].inputSize = (6, 6)
layerParams[0].layerSize = (6, 6)
#layerParams[1].inputSize = (6, 6)
#layerParams[1].layerSize = (4, 4)

hierarchy = hth.HTFERL_Hierarchy(layerParams, -0.1, 0.1)

#sequence = [[0.0, 1.0, 1.0, 1.0],
#            [0.0, 1.0, 1.0, 0.0],
#            [0.0, 1.0, 0.0, 1.0],
#            [0.0, 1.0, 1.0, 1.0],
#            [1.0, 0.0, 0.0, 0.0],
#            [0.0, 1.0, 1.0, 1.0],
#            [1.0, 1.0, 0.0, 0.0]]

sequence = [[1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0]]

for i in range(0, 1000):
    for j in range(0, len(sequence)):
        hierarchy.setInput((1, 1), sequence[j][0])
        hierarchy.setInput((2, 1), sequence[j][1])
        hierarchy.setInput((3, 1), sequence[j][2])
        hierarchy.setInput((4, 1), sequence[j][3])

        hierarchy.activateFeedForward()

        hierarchy.activateFeedBack()

        hierarchy.learn(0.01, 0.01)

        print(str(hierarchy.getReconstruction((1, 1))) + " ")
        print(str(hierarchy.getReconstruction((2, 1))) + " ")
        print(str(hierarchy.getReconstruction((3, 1))) + " ")
        print(str(hierarchy.getReconstruction((4, 1))) + " ")
        print("\n")

        hierarchy.stepEnd()
    