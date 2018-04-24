import gym
from tensorflow import keras
import numpy as np
import os
from differential_evolution import DifferentialEvolution as DE

### https://github.com/openai/gym/wiki/CartPole-v0

SEED = 42
MAX_REWARD = 200

def create_model(x, hidden, y):
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(x, activation='relu', input_dim=x, bias_initializer = 'glorot_uniform'))
    for layer in hidden:
        model.add(keras.layers.Dense(layer, activation='relu', bias_initializer = 'glorot_uniform'))
    #model.add(keras.layers.Dense(y, activation='relu', bias_initializer = 'glorot_uniform'))
    model.add(keras.layers.Dense(y * 2, activation='softmax', bias_initializer = 'glorot_uniform'))

    #model.add(keras.layers.Dense(x, input_dim=1, activation='relu'))
    #for layer in hidden:
    #    model.add(keras.layers.Dense(layer, activation='relu', bias_initializer='glorot_uniform'))
    #model.add(keras.layers.Dense(y, activation='softmax'))
    return model

def output_to_action(x):
    # relu
    #return 0 if x[0][0] < 0.5 else 1

    # softmax
    return np.argmax(x)

def check_correct_model(env, model):
    observation = env.reset()
    assert observation.shape == (4,)

    observation = observation.reshape(1,4)
    assert observation.shape == (1,4)


    result = model.predict(observation, 1)
    # relu
    #assert result.shape == (1, 1)
    # softmax
    assert result.shape == (1, 2)

def get_random_weights():
    shapes = list(map(lambda w: w.shape, weights))
    new_weights = []
    for shape in shapes:
        new_weights.append(np.random.random_sample(shape) * 2 - 1)
    return new_weights

def set_random_weights(model):
    new_weights = get_random_weights()
    model.set_weights(new_weights)

def save_model(model, score):
    def get_unique_filepath():
        filepath = 'CartPoleModels/CartPoleModel_{}'.format(score)

        if not os.path.exists(filepath):
            return filepath
        
        def get_filepath_with_id(id):
            return '{}_{}'.format(filepath, id)

        i = 0
        while os.path.exists(get_filepath_with_id(i)):
            i += 1

        return get_filepath_with_id(i)

    filepath = get_unique_filepath()
    model.save(filepath)

def load_model(score, id=None):
    filepath = 'CartPoleModels/CartPoleModel_{}'.format(score)
    if id is not None:
        filepath = '{}_{}'.format(filepath, id)
    return keras.models.load_model(filepath)

def reshape_weights_for_de(weights):
    D = sum(map(lambda w: w.size, weights))
    result = np.empty(D)
    i = 0
    for w in weights:
        result[i:i + w.size] = w.flatten()
        i += w.size
    return result

def reshape_weights_for_keras(flatten_weights):
    result = []
    i = 0
    for w in weights:
        result.append(flatten_weights[i:i + w.size].reshape(w.shape))
        i+= w.size
    return result

def get_inititial_population(size):
    population = range(size)
    population = map(lambda _: get_random_weights(), population)
    population = map(lambda w: reshape_weights_for_de(w), population)
    return list(population)

def run_episode(model, weights):
    model_weights = reshape_weights_for_keras(weights)
    model.set_weights(model_weights)

    env.seed(SEED)
    observation = env.reset()
    score = 0.0
    for t in range(200):
        #env.render()
        result = model.predict(observation.reshape(1,4), 1)
        action = output_to_action(result)
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    if score >= 15:
        print("score=", score)
    return score

def run_episodes(model):
    scores = []
    for i in range(100):
        #env.seed(SEED)
        observation = env.reset()
        score = 0.0
        for t in range(MAX_REWARD):
            env.render()
            result = model.predict(observation.reshape(1,4), 1)
            action = output_to_action(result)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print('Iteration={}, Score={}'.format(i, score))
                scores.append(score)
                break
    mean = np.mean(scores)
    print('Min={}, max={}, mean={}'.format(np.min(scores),np.max(scores), mean))
    return mean

env = gym.make('CartPole-v0')
print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

load = False
if load:
    model = load_model(123.58)
    p = run_episodes(model)
    exit()
else:
    X_DIM = 4
    Y_DIM = 1
    HIDDEN_LAYERS = [4,4,4]
    model = create_model(X_DIM, HIDDEN_LAYERS, Y_DIM)
    model.compile(optimizer='sgd', loss='binary_crossentropy')

check_correct_model(env, model)
weights = model.get_weights()

#run_episodes(model, False, False)
#de_weights = reshape_weights_for_de(weights)
#keras_weights = reshape_weights_for_keras(de_weights, weights)

#run_episodes(model, update_weights = False, save_model = False)

#best = run_episodes(model, update_weights = False, save_model = False)
#run_episodes(best, update_weights = False, save_model = False)

de = DE(lambda params: run_episode(model, params))
population = get_inititial_population(50)
params = de.run(population, 50)

model.set_weights(reshape_weights_for_keras(params))
mean = run_episodes(model)
#if mean >= 195.0:
save_model(model, mean)