import gym
import numpy as np
import os

import keras.backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import load_model

def discount_rewards(rewards, discount_factor=0.99):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * discount_factor + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def get_action(model, state, action_size):
    policy = model.predict(state, batch_size=1).flatten()
    return np.random.choice(action_size, 1, p=policy)[0]

def makeOptimizer(model, action_size, learning_rate=0.001):
    action = K.placeholder(shape=[None, action_size])
    discounted_rewards = K.placeholder(shape=[None, ])

    good_prob = K.sum(action * model.output, axis=1)
    eligibility = K.log(good_prob) * discounted_rewards
    loss = -K.sum(eligibility)

    optimizer = Adam(lr=learning_rate)
    updates = optimizer.get_updates(model.trainable_weights, [], loss)
    train = K.function([model.input, action, discounted_rewards], [], updates=updates)
    optimizer = train
    return train

def trainOnEpisodes(opt, states, actions, rewards):
    discounted_rewards = discount_rewards(rewards)
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    opt([states, actions, discounted_rewards])
    return

def eval_cur_policy(env, model):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    min_score = np.inf
    max_score = -np.inf
    scores = []
    for _ in xrange(100):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        eps_step = 0

        while not done:
            action = get_action(model, state, action_size)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            eps_step += 1

            if eps_step > 199:
                done = True

            score += reward
            state = next_state

        scores.append(score)
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score
    return np.mean(scores), min_score, max_score

def reinforce(env, reinforce_model, reset_in_steps=3):
    with open('logs.txt', "a") as logs:
        state_size = env.observation_space.shape
        action_size = env.action_space.n
        print "Action size: {}".format(action_size)

        opt = makeOptimizer(reinforce_model, action_size)

        states, actions, rewards = [], [], []
        reset = 0
        scores, max_scores, min_scores, eps = [], [], [], []

        total_rewards = []
        
        for e in xrange(600):
            done = False
            score = 0

            state = env.reset()
            state = np.reshape(state, [1, state_size])

            while not done:
                # Perform hard resets of session to avoid memory leak errors
                if reset % reset_in_steps == 0:
                    reinforce_model.save("reinforce_model.h5")
                    K.clear_session()
                    reinforce_model = load_model("reinforce_model.h5")
                    opt = makeOptimizer(reinforce_model, action_size)
                    reset = 0
                else:
                    reset += 1


                action = get_action(reinforce_model, state, action_size)
                print "Action: {}".format(action)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                states.append(state[0])
                rewards.append(reward)

                #Log rewards
                logs.write(str(reward))

                act = np.zeros(action_size)
                act[action] = 1
                actions.append(act)

                state = next_state

                if done:
                    trainOnEpisodes(opt, states, actions, rewards)
                    total_rewards.append(np.mean(rewards))
                    states, actions, rewards = [], [], []

    return total_rewards
