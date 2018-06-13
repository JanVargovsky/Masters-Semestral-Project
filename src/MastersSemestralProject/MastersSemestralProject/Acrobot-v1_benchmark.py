import gym
import benchmark_runner
import agent_ai
import numpy as np
from differential_evolution import de, de_jade_with_archive, de_shade

env = gym.make('Acrobot-v1')

model = agent_ai.create_model(6, [3,3], 3, 'softmax')
weights = model.get_weights()
NP = 20
D = sum(map(lambda w: w.size, weights))
N = 5 # must be 30+

def run_env_for_fitness(flatten_weights):
    model_weights = agent_ai.reshape_weights_for_keras(flatten_weights, weights)
    model.set_weights(model_weights)

    #env.seed(seed)
    observation = env.reset()
    score = 0.0
    while True:
        env.render()
        result = model.predict(observation.reshape(1,6), 1)
        action = agent_ai.output_to_action(result)
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break

    if score > -500:
        print("score={}".format(score))
    return score   

initial_population = agent_ai.get_inititial_population(NP, weights)
end_condition = lambda _, g: g >= 30
results = []
result_to_value = lambda de_result: de_result.generations
for (name, alg) in [('SHADE', de_shade), ('JADE', de_jade_with_archive), ('DE', de)]:
#for (name, alg) in [('JADE', de_jade_with_archive)]:
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
    filename = 'results/Acrobot={}_runs={}'.format(name, N)
    with open(filename, 'w') as fw:
        fw.write(content)