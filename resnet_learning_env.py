# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import numpy
import resnet50_training


class Resnet01Env(Env):
    """Implements a Residual network training enviornment.
    This version only allows identity or 0 connections. The states
    represent the last 6 trainable layers. The first 6 trainable
    layers are left untrainable to improve training speed.
    Parameters
    ----------
    fine_tune_all: bool
      Whether to fine tune the network on all layers or top 6.
    Attributes
    ----------
    P: environment model
    """
    def __init__(self, fine_tune_all=False):
        self.action_space = spaces.Discrete(6*2)
        self.observation_space = spaces.MultiDiscrete(
            [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])

        #initial state
        self._reset()

        self.fine_tune_all = fine_tune_all

    def update_model(self):
        """Sets the trainable model according to the current state
        Returns
        -------
        model
          A Keras trainable model
        """
        model = resnet50_training.init_compile_model(self.state)
        return model

    def _reset(self):
        """Reset the environment.
        The start state is the the original Resnet architecture.
        Returns
        -------
        [int, int, int, int, int, int]
          A list representing the current state with meanings
          [layer0_type, layer1_type, layer2_type,
           layer3_type, layer4_type, layer5_type]
        """
        self.state = [1, 1, 1, 1, 1, 1]
        self.model = self.update_model()
        return self.state

    def _step(self, action):
        """Execute the specified action.
        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.
        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        layer_to_change = int(action) / 2
        change_to_identity = action % 2
        self.state[layer_to_change] = change_to_identity
        self.model = self.update_model()

        #train the model
        train_history = resnet50_training.fine_tune_model(self.model)

        #compute the reward
        print(train_history)
        #rewards = validation acc or top 5 val accuracy

        return tuple(self.state), reward, is_terminal, None

    def _render(self, mode='human', close=False):
        print(self.state)

    def _seed(self, seed=None):
        """Set the random seed.
        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        numpy.random.seed(seed)

register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})