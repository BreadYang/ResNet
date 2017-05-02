import tensorflow as tf
import numpy as np

import resnet_learning_env as renv

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Activation

def makeReinforceModel(num_layers=2):
    """
    num_layers: The number of choices of layer types to make
                each residual connection
    """
    model = Sequential()
    model.add(Dense(128, input_dim=6))
    model.add(Activation('relu'))
    model.add(Dense(num_layers*6))
    model.add(Activation('softmax'))
    return model

def loadReinforceModel():
    return load_model("reinforce_model.h5")

#Set up the agent and enviornment
#Our agent is a simple 1 hidden layer NN
reinforce_model = makeReinforceModel()
env = renv.Resnet01Env()

import reinforce

#Run reinforcement learning
reinforce.reinforce(env, reinforce_model)
