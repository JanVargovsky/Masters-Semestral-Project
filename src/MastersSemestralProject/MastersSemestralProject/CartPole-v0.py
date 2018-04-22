import gym
from tensorflow import keras
import numpy as np
import os

### https://github.com/openai/gym/wiki/CartPole-v0

def create_model(x, hidden, y):
    model = keras.models.Sequential()
    #model.add(keras.layers.Dense(1, input_dim=X_DIM, activation='relu', ))
    model.add(keras.layers.Dense(x, input_dim=1))
    for layer in hidden:
        model.add(keras.layers.Dense(layer, activation='relu', bias_initializer = 'glorot_uniform'))
    model.add(keras.layers.Dense(y * 2, activation='softmax'))
    return model

def output_to_action(x):
    return np.argmax(x[0])

def set_random_weights(model):
    shapes = list(map(lambda w: w.shape, model.get_weights()))

    new_weights = []
    for shape in shapes:
        new_weights.append(np.random.random_sample(shape) * 2 - 1)

    model.set_weights(new_weights)

def save_model(model, score):
    def get_unique_filepath():
        filepath = 'CartPoleModels/CartPoleModel_{}'.format(score)

        if not os.path.exists(filepath):
            return filepath
        
        def get_filepath_with_id(id):
            return '{}_{}'.format(filepath, id)

        i = 0
        while os.path.exists(get_filepath_with_id(id)):
            i += 1

        return get_filepath_with_id(id)

    filepath = get_unique_filepath()
    model.save(filepath, False)

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
        result[i:i+w.size] = w.flatten()
        i += w.size
    return result

def reshape_weights_for_keras(flatten_weights, weights):
    result = []
    i = 0
    for w in weights:
        result.append(flatten_weights[i:i+w.size].reshape(w.shape))
        i+= w.size
    return result

def run_episodes(model, update_weights, save_model):
    scores = []
    for i in range(2000):
        observation = env.reset()
        score = 0.0
        for t in range(200):
            result = model.predict(observation, 1)
            action = output_to_action(result)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                if score >= 15:
                    print('Iteration={}, Score={}'.format(i, score))
                if save_model and score >= 50:
                    save_model(model, score)
                if not scores or (scores and max(scores) < score):
                    print('New best score {}'.format(score))
                    best = model
                scores.append(score)
                set_random_weights(model)
                break
    print('Best score={}, mean={}'.format(np.max(scores), np.mean(scores)))
    return best

env = gym.make('CartPole-v0')
print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

load = False
if load:
    model = load_model(200.0)
else:
    X_DIM = 4
    Y_DIM = 1
    HIDDEN_LAYERS = [8,8,8]
    model = create_model(X_DIM, HIDDEN_LAYERS, Y_DIM)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#weights = model.get_weights()
#de_weights = reshape_weights_for_de(weights)
#keras_weights = reshape_weights_for_keras(de_weights, weights)

best = run_episodes(model, update_weights = False, save_model = False)
run_episodes(best, update_weights = False, save_model = False)