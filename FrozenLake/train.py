import gymnasium as gym
import numpy as np
import pickle

# 1. Setup Environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
Qtable = np.zeros((env.observation_space.n, env.action_space.n))

# 2. Hyperparameters
total_episodes = 10000
learning_rate = 0.7
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

print("🚀 Training started...")

for episode in range(total_episodes):
    state, _ = env.reset()
    done = False
    
    for step in range(100):
        # Epsilon-greedy: Choose random action or best known action
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Qtable[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)

        # Update Q-table using Bellman Equation
        Qtable[state, action] = Qtable[state, action] + learning_rate * (
            reward + gamma * np.max(Qtable[new_state, :]) - Qtable[state, action]
        )
        
        state = new_state
        if terminated or truncated:
            break
            
    # Reduce epsilon to explore less over time
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

# Save the successful Q-table
with open("qtable.pkl", "wb") as f:
    pickle.dump(Qtable, f)

print("✅ Training complete! Q-table saved.")