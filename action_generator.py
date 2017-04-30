# coding: utf-8
"""Implementation of the action generator for stepwise model selection for ResNet"""
import numpy as np
import build_CNN as CNN

class ActionGen:
    """Implementation of the action generator.

    Parameters
    ----------
    metwork_dim: int
      Dimension of action space, or the number of residual connections.
    choice_count: int
      Number of choices tested in each epoch (test the # best from previous epoch).
    accuracyDic: Dictionary
      Dictionary storing all previous accuracy
    prefix: array
      A predetermined part of residual connection.
    """

    def __init__(self, choice_count, network_dim=12, prefix=[1, 1, 1, 1, 1, 1]):
        self.action_dim = int(network_dim) - len(prefix)
        self.choice_count = int(choice_count)
        self.accuracyDic = {}
        self.prefix = prefix
        self.action_space = [1, 2, 3, 4]
        self.prefixLen = len(prefix)
        self.bestAction = np.zeros(self.action_dim, dtype=np.int)
        self.bestAccuracy = 0.0
        for i in range(self.action_dim):
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
        """get list of actions of size target_dim, i.e, there are target_dim numbers of residual connections in the
        network
        target_dim: int
            Number of dimensions in the non-prefix part being non-zero.
        """
        assert target_dim <= self.action_dim
        output = []
        if target_dim == 1:
            zeros = np.zeros(self.action_dim, dtype=np.int)
            for i in range(self.action_dim):
                tmp = np.copy(zeros)
                for residual in self.action_space:
                    tmp[i] = residual
                    output.append(tmp.copy())
        else:
            action_dict = self.accuracyDic[target_dim - 1]
            sortedDic = sorted(action_dict.items(), key=lambda K: K[1][0] / float(K[1][1]), reverse=True)
            cnt = 0
            for key, value in sortedDic:
                if cnt == self.choice_count:
                    break
                cnt += 1
                key_a = self.num2array(key)  ##  key in array
                for index in range(self.action_dim):
                    if key_a[index] == 0: ## connection not tried
                        for residual in self.action_space:
                            _action = np.copy(key_a)
                            _action[index] = residual
                            if self.array2num(_action) not in self.visited:
                                self.visited[self.array2num(_action)] = 1
                                output.append(list(_action.copy()))

        assert len(output) > 0
        output_w_prefix = []
        for action in output:
            tmpoutput = list(np.copy(self.prefix))
            tmpoutput += list(action)
            tmpoutput = add_conv_connection(tmpoutput)
            output_w_prefix.append(list(tmpoutput.copy()))
        return list(output_w_prefix)

    def storeAccuracy(self, action, accuracy):
        """ Store the accuracy of the action into dictionary
        action: array full action
        accuaracy: float
        """
        action = rm_conv_connection(action)
        assert max(action) <= 5
        action = action[self.prefixLen:]
        dimension = np.count_nonzero(action)
        testcnt = 1
        if self.array2num(action) in self.accuracyDic[dimension]:
            testcnt = self.accuracyDic[dimension][self.array2num(action)][1] + 1
        self.accuracyDic[dimension][self.array2num(action)] = [np.float(accuracy), 1]
        return self.accuracyDic[dimension][self.array2num(action)]

    def array2num(self, array):
        num = 0
        for i in array:
            num = int(num * 10) + i
        return num

    def num2array(self, numin):
        num = int(numin)
        array = np.zeros(self.action_dim, dtype=np.int)
        index = self.action_dim - 1
        while num != 0:
            array[index] = num % 10
            num = int(num / 10)
            index -= 1
        return array

def find_last_changed(state1, state2):
    assert len(state1) == len(state2)
    i = 0
    for i in range(len(state2)):
        if state2[i] != state1[i]:
            break
    return i


## from 12 to 16
def add_conv_connection(state):
    stateold = np.copy(state)
    state = np.zeros(16)
    state[0] = 1
    state[1:3] = stateold[0:2]
    state[3] = 1
    state[4:7] = stateold[2:5]
    state[7] = 1
    state[8:13] = stateold[5:10]
    state[13] = 1
    state[14:16] = stateold[10:12]
    state = [int(i) for i in state]
    return np.asarray(state)


## from 16 to 12
def rm_conv_connection(state):
    stateold = np.copy(state)
    state = np.zeros(12)
    state[0:2] = stateold[1:3]
    state[2:5] = stateold[4:7]
    state[5:10] = stateold[8:13]
    state[10:12] = stateold[14:16]
    state = [int(i) for i in state]
    return np.asarray(state)
    ############

## Test the top three resNet from the previous epoch, with five 1 as prefix
generator = ActionGen(3, 12, [1, 1, 1, 1, 1])
for dim in range(6):
    actions = generator.getActions(dim+1)
    for action in actions:
        # print(action)
        # model = CNN.update_model(action, "imagenet")
        # accuacy = model.fit
        accuacy = 0.5
        generator.storeAccuracy(action, accuacy)
