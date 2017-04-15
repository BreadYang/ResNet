# coding: utf-8
"""Implementation of the action generator for stepwise model selection for ResNet"""
import numpy as np

class ActionGen:
    """Implementation of the action generator.

    Parameters
    ----------
    action_dim: int
      Dimension of action space, or the number of residual connections.
    choice_count: int
      Number of choices tested in each epoch (test the # best from previous epoch).
    accuracyDic: Dictionary
      Dictionary storing all previous accuracy
    """

    def __init__(self, Choice_num, action_dim=16):
        self.action_dim = int(action_dim)
        self.choice_count = int(Choice_num)
        self.accuracyDic = {}
        self.bestAction = np.zeros(self.action_dim, dtype=np.int)
        self.bestAccuracy = 0.0
        for i in range(action_dim):
            index = i + 1
            self.accuracyDic[index] = {}
        self.visited = {}

    def _reset(self):
        """Reset the memory.
        """
        self.accuracyDic = {}
        self.bestAction = np.zeros(self.action_dim, dtype=np.int)
        self.bestAccuracy = 0.0
        for i in range(self.action_dim):
            index = i + 1
            self.accuracyDic[index] = {}
        self.visited = {}

    def getActions(self, target_dim):
        """get list of actions of size current_dim, i.e, there are current_dim of digits being one
        """
        visited = {}
        assert target_dim <= self.action_dim
        output = []
        if target_dim == 1:
            zeros = np.zeros(self.action_dim, dtype=np.int)
            for i in range(self.action_dim):
                tmp = np.copy(zeros)
                tmp[i] = 1
                output.append([tmp])
        else:
            action_dict = self.accuracyDic[target_dim - 1]
            print action_dict.keys()
            sortedDic = sorted(action_dict.items(), key=lambda (k, (v1, v2)): v1/float(v2), reverse=True)
            cnt = 0
            for key, value in sortedDic:
                if cnt == self.choice_count:
                    break
                cnt += 1
                key_a = self.num2array(key) ## key in array
                for index in range(self.action_dim):
                    if key_a[index] != 1:
                        tmp_a = np.copy(key_a)
                        tmp_a[index] = 1
                        if self.array2num(tmp_a) not in visited:
                            visited[self.array2num(tmp_a)] = 1
                            output.append([tmp_a])
        assert len(output) > 0
        return output

    def storeAccuracy(self, action, accuracy):
        assert max(action) == 1
        dimension = sum(action)
        self.accuracyDic[dimension][self.array2num(action)] = [np.float(accuracy), 1]
        return self.accuracyDic[dimension][self.array2num(action)]

    def array2num(self, array):
        num = 0
        for i in array:
            num = num*10 + i
        return num

    def num2array(self, numin):
        num = int(numin)
        array = np.zeros(self.action_dim, dtype=np.int)
        index = self.action_dim-1
        while num != 0:
            array[index] = num % 10
            num /= 10
            index -= 1
        return array


## Test
generator = ActionGen(1, 5)
generator.getActions(1)
print generator.storeAccuracy([0,0,1,0,0], 0.5)
print generator.getActions(2)
