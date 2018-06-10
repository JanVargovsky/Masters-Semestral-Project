from tensorflow import keras
import numpy as np

def create_model(x, hidden, y):
    model = keras.models.Sequential()

    #model.add(keras.layers.Dense(x, activation='relu', input_dim=x, bias_initializer = 'glorot_uniform'))
    model.add(keras.layers.InputLayer((x,)))
    #model.add(keras.layers.Permute((2,1), input_shape=(4,1)))
    #model.add(keras.layers.Input((x,1)))
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

def get_random_weights(weights):
    shapes = list(map(lambda w: w.shape, weights))
    new_weights = []
    for shape in shapes:
        new_weights.append(np.random.random_sample(shape) * 2 - 1)
    return new_weights

def reshape_weights_for_de(weights):
    D = sum(map(lambda w: w.size, weights))
    result = np.empty(D)
    i = 0
    for w in weights:
        result[i:i + w.size] = w.flatten()
        i += w.size
    return result

def reshape_weights_for_keras(flatten_weights, weights):
    result = []
    i = 0
    for w in weights:
        result.append(flatten_weights[i:i + w.size].reshape(w.shape))
        i+= w.size
    return result

def get_inititial_population(size, weights):
    population = range(size)
    population = map(lambda _: get_random_weights(weights), population)
    population = map(lambda w: reshape_weights_for_de(w), population)
    return list(population)