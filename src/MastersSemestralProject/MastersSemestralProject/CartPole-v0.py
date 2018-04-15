import gym
import tensorflow as tf
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

### https://github.com/openai/gym/wiki/CartPole-v0

X_DTYPE = tf.float32
X_DIM = 4

HIDDEN_LAYERS = [4,4,4]

Y_DTYPE = tf.int64
Y_DIM = 1

def create_mlp():
    x = tf.placeholder(X_DTYPE, [None, X_DIM], name='x')

    hidden_layers = []
    last_layer = x
    for index, layer in enumerate(HIDDEN_LAYERS):
        hidden_layer = tf.layers.dense(last_layer, layer, activation= tf.nn.relu, name='l-{}'.format(index))
        last_layer = hidden_layer
        hidden_layers.append(last_layer)

    y = tf.layers.dense(last_layer, Y_DIM, activation= tf.nn.relu, name='y')

    return (x, hidden_layers, y)

def discrete_output(x):
    return 0 if x < 0 else 1

env = gym.make('CartPole-v0')
print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

(x, hidden_layers, y) = create_mlp() 

init = tf.global_variables_initializer()
sess = tf.Session()

with sess.as_default():
    sess.run(init)

    scores = []
    for i_episode in range(20):
        observation = env.reset()
        score = 0
        for t in range(200):
            env.render()
            #print(observation)
            #action = env.action_space.sample()
            input = observation.reshape(1,4)
            result = sess.run(y, feed_dict = {x: input })
            action = discrete_output(result[0][0])
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print('Episode finished after {} timesteps, score={}'.format(t + 1, score))
                scores.append(score)
                break