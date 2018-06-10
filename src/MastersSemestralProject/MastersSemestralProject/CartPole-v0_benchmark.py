import gym
import benchmark_runner
import agent_ai
import numpy as np
from differential_evolution import de, de_jade_with_archive, de_shade

env = gym.make('CartPole-v0')

model = agent_ai.create_model(4, [2,2], 1)
weights = model.get_weights()
NP = 50
D = sum(map(lambda w: w.size, weights))
MAX_REWARD = 200
N = 30 # must be 30+
SEED_COUNT = 10 # should be 30

seeds = [42 + i for i in range(SEED_COUNT)] 
def run_env_for_fitness(flatten_weights):
    model_weights = agent_ai.reshape_weights_for_keras(flatten_weights, weights)
    model.set_weights(model_weights)

    scores = []
    for seed in seeds:
        env.seed(seed)
        observation = env.reset()
        score = 0.0
        while True:
            result = model.predict(observation.reshape(1,4), 1)
            action = agent_ai.output_to_action(result)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                scores.append(score)
                break

    score = np.mean(scores)
    if score >= 50:
        print("mean={:0.1f}, scores={}".format(score, scores))
    return score
     

initial_population = agent_ai.get_inititial_population(NP, weights)
for reward in [2500, 1000, 500]:
    MAX_REWARD = reward
    end_condition = lambda reward: reward >= MAX_REWARD
    env._max_episode_steps = MAX_REWARD
    results = []
    result_to_value = lambda de_result: de_result.generations
    for (name, alg) in [('SHADE', de_shade), ('JADE', de_jade_with_archive), ('DE', de)]:
        action = lambda: alg(run_env_for_fitness, initial_population, end_condition, 'max')
        print("Start {}".format(name))
        result = benchmark_runner.run(N, action, result_to_value)
        print("End {}".format(name))
    
        filename = 'results/max-reward={}_alg={}_runs={}_seeds={}'.format(MAX_REWARD, name, N, SEED_COUNT)
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
        with open(filename, 'w') as fw:
            fw.write(content)