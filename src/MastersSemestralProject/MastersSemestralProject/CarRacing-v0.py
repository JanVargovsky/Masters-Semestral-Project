import gym
import gym.envs.box2d.car_racing

gym.envs.box2d.car_racing.WINDOW_W = 600
gym.envs.box2d.car_racing.WINDOW_H = 500
env = gym.make("CarRacing-v0")

for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break