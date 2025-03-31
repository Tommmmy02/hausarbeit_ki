import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchPickAndPlaceDense-v3', render_mode="human", max_episode_steps=1000)

observation, info = env.reset()

while True:
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated