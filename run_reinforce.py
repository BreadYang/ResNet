import tensorflow as tf
import numpy as np

import resnet_learning_env as renv
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Activation

from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("train_data_dir", '',
                     "Path to training data for Resnet.")
flags.DEFINE_string("val_data_dir", '',
                     "Path to validation data for Resnet.")
flags.DEFINE_bool("save_features", False,
                  "Whether to save bottleneck features before training. Only do this once")
flags.DEFINE_integer("epochs", 20, "The number of epochs to fine tune for")
flags.DEFINE_bool("soft_start", False, "Whether to penalize doing the same action twice.")

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

if __name__ == "__main__":
    #Save bottleneck features to save on computation time.
    if FLAGS.save_features:
        save_bottleneck_features(train_data_dir=FLAGS.train_data_dir,
                                val_data_dir=FLAGS.val_data_dir, 
                                overwrite=False)

    #Set up the agent and enviornment
    #Our agent is a simple 1 hidden layer NN
    reinforce_model = makeReinforceModel()
    env = renv.Resnet01Env(fine_tune_epochs=FLAGS.epochs)

    import reinforce

    #Run reinforcement learning
    avg_rewards = reinforce.reinforce(env, reinforce_model, soft_start=FLAGS.soft_start)

    #Plotting performance over time
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Validation Accuracy")

    #Save the best learned Resnet model
    env.save_model('best_model.h5')
