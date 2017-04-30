# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import numpy
import build_CNN


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
    nS: number of states
    nA: number of actions
    P: environment model
    """
    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 0
        self.nA = 0

        #initial state
        self._reset()

    def update_model(self):
        model = build_CNN.update_model(self.state, 'imagenet')
        return model

    def _reset(self):
        """Reset the environment.
        The server should always start on Queue 1.
        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        self.state = [1,1,1,1,1,1]
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
        layer_to_change = action / 6
        change_to_identity = action % 3
        self.state[layer_to_change] = change_to_identity
        self.model = self.update_model()

        #train the model

        #compute the reward
        #rewards = validation acc or top 5 val accuracy

        return tuple(self.state), reward, is_terminal, None

    def _render(self, mode='human', close=False):
        print self.state

    def _seed(self, seed=None):
        """Set the random seed.
        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        numpy.random.seed(seed)

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.
        This should be in the same format at the provided environments
        in section 2.
        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.
        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        def all_possible_delta(n):
            if (n == 1):
                return [[0],[1]]
            else:
                r = all_possible_delta(n-1)
                l = []
                for i in r:
                    l.append(i + [0])
                    l.append(i+ [1])
                return l
        def prob_delta_state(delta_state, probs):
            prob = 1
            for i,d in enumerate(delta_state):
                if d == 0:
                    prob *= (1-probs[i])
                else:
                    prob *= (probs[i])
            return prob

        state = list(state)
        max_limit = 5
        is_terminal = False
        newstate, reward = self.do_action_only(action, state)
        probs = [self.p1,self.p2,self.p3]
        possible_delta = all_possible_delta(3)
        possible_states = [[newstate[0], newstate[1]+a,newstate[2]+b,newstate[3]+c]for a,b,c in possible_delta]
        prob_possible_states = [prob_delta_state(ds, probs) for ds in possible_delta]
        outcomes = [(pr, st, reward, is_terminal) for (pr,st) in zip(possible_states,prob_possible_states)]
        

        return outcomes

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