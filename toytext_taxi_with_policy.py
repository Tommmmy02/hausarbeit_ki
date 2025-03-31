import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

alpha = 0.1  
gamma = 0.9   
epsilon = 1.0 
epsilon_min = 0.1
epsilon_decay = 0.99
episodes = 1000
rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(q_table[state, :])  
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        print(f"Episode {episode}, Zustand {state}, Aktion {action}, Reward {reward}")
    
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        state = next_state
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

env.close()

print("Trainiertes Q-Table:")
print(q_table)

env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state, :])
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Gesamtbelohnung")
plt.title("Lernfortschritt von Q-Learning")
plt.show()
