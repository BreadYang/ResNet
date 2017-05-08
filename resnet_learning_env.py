# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import numpy
import copy
import resnet50_training
import build_CNN

# CHANGE THIS TO YOUR DIRECTORY
train_data_dir = 'tiny-imagenet-200/train'
val_data_dir = 'tiny-imagenet-200/val'
# pretrained_imagenet = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_imagenet = 'top_half_weights.f5'


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
    def __init__(self, fine_tune_all=False, max_eps_step=5, fast_split_model=True,
                    pretrained_weights=pretrained_imagenet,
                    allow_continuous_finetuning=False):
        self.allow_continuous_finetuning = allow_continuous_finetuning
        self.action_space = spaces.Discrete(6*2)
        self.observation_space = spaces.MultiDiscrete(
            [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])
        self.fast_split_model = fast_split_model
        
        #Initial state is all identity connections
        self.state = [1, 1, 1, 1, 1, 1]
        self.last_reward = 0
        self.fine_tune_all = fine_tune_all
        self.max_eps_step = max_eps_step


        if self.fast_split_model:
            build_CNN.save_bottleneck_features(train_data_dir=train_data_dir, val_data_dir=val_data_dir, overwrite=False)
            self.model = build_CNN.top_model(self._convert_state_to_internal_state(self.state), weights_path=pretrained_weights)

        else:
            self.model = resnet50_training.init_compile_model(self.state)
    def update_model(self):
        """Sets the trainable model according to the current state
        Returns
        -------
        model
          A Keras trainable model
        """
        model = None
        if not self.fast_split_model:
            model = resnet50_training.init_compile_model(self.state)
        else:
            model = build_CNN.top_model(self._convert_state_to_internal_state(self.state))
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
        old_state = []
        old_state = self.state
        self.state = [1, 1, 1, 1, 1, 1]
        if old_state != self.state:
            self.model = self.update_model()
        self.eps_step = 0
        return self.state


    def _convert_state_to_internal_state(self, state):
        initial_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        do_not_alter = [0,3,7,13]
        num_states = len(initial_state)
        index = 0
        for i in xrange(num_states):
            if i >= 9:
                if i in do_not_alter:
                    continue
                initial_state[i] = state[index]
                index += 1

        return initial_state

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
        old_state = copy.copy(self.state)
        self.state[layer_to_change] = change_to_identity

        if self.allow_continuous_finetuning or self.state != old_state:
            self.model = self.update_model()
            print("State is " + str(self.state))
            #train the model
            train_history = None
            if self.fast_split_model:
                train_history = build_CNN.finetune_top_model(self.model)
            else:
                train_history = resnet50_training.fine_tune_model(self.model)
            #compute the reward
            hist = train_history.history
            reward = hist['val_acc'][-1]
            self.last_reward = reward
            self.eps_step += 1
            is_terminal = self.eps_step >= self.max_eps_step

            return tuple(self.state), reward, is_terminal, None
        else:
            is_terminal = self.eps_step >= self.max_eps_step
            return tuple(self.state), self.last_reward, is_terminal, None

    def _render(self, mode='human', close=False):
        print(self.state)

    def save_model(self,filepath):
        self.model.save(filepath)

    def _seed(self, seed=None):
        """Set the random seed.
        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        numpy.random.seed(seed)
