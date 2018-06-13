import gym
import benchmark_runner
import agent_ai
import numpy as np
from differential_evolution import de, de_jade_with_archive, de_shade

env = gym.make('MountainCarContinuous-v0')

model = agent_ai.create_model(2, [2,2], 1, 'tanh')
weights = model.get_weights()
NP = 50
D = sum(map(lambda w: w.size, weights))
REWARD = 100 + 90
N = 5 # must be 30+

def run_env_for_fitness(flatten_weights):
    model_weights = agent_ai.reshape_weights_for_keras(flatten_weights, weights)
    model.set_weights(model_weights)

    env.seed(42)
    observation = env.reset()
    score = 0.0
    topLeft = topRight = observation[0]

    i = 0
    while True:
        #env.render()
        #print(observation)

        action = model.predict(observation.reshape(1,2), 1)
        #print(action)
        observation, reward, done, info = env.step(action)
        score += reward
        try:
            topLeft = min(topLeft, observation[0][0])
            topRight = max(topRight, observation[0][0])
        except :
            print("FAIL: {}".format(observation))
        
        if done:
            break

    print("left={:0.2f}, right={:0.2f}, score={:0.2f}".format(topLeft, topRight, score))
    return 100 + score if score >= 0 else topRight
     

initial_population = agent_ai.get_inititial_population(NP, weights)
end_condition = lambda reward, g: reward >= REWARD or g >= 100
results = []
result_to_value = lambda de_result: de_result.generations
for (name, alg) in [('SHADE', de_shade), ('JADE', de_jade_with_archive), ('DE', de)]:
#for (name, alg) in [('SHADE', de_shade), ('JADE', de_jade_with_archive)]:
#for (name, alg) in [('DE', de)]:
    action = lambda: alg(run_env_for_fitness, initial_population, end_condition, 'max')
    print("Start {}".format(name))
    result = benchmark_runner.run(N, action, result_to_value)
    print("End {}".format(name))
    
    export = []
    export.append('Generations:')
    export.append(str(result.Result))
    export.append('')
    export.append('Time:')
    export.append(str(result.Time))
    export.append('')
    export.append('Required generations + fitness history:')
    for de_result in result.Results:
        export.append('{}, {}'.format(str(de_result.generations), str(de_result.history)))
    export.append('')
    export.append('Times:')
    export.append(str(result.Times))
    export.append('')
    export.append('Params:')
    for de_result in result.Results:
        export.append(str(de_result.params))

    content = '\n'.join(export) + '\n'
    print(content)
    filename = 'results/MountainCarContinuous_alg={}_runs={}_population-size={}'.format(name, N, NP)
    with open(filename, 'w') as fw:
        fw.write(content)