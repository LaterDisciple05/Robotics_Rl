import gymnasium as gym
import numpy as np
import random
import time
import pickle

# --- 1. INITIALIZE ENVIRONMENT ---
# In VS Code, we use "human" to see a real popup window
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

state_size = env.observation_space.n
action_size = env.action_space.n
qtable = np.zeros((state_size, action_size))

# --- 2. TRAINING PARAMETERS ---
total_episodes = 5000        
learning_rate = 0.7          
gamma = 0.95                 
epsilon = 1.0                
max_epsilon = 1.0            
min_epsilon = 0.01           
decay_rate = 0.005           

print("🚀 Training... (The window might stay black during training to save speed)")

# Turn off rendering during training to make it 100x faster
env.unwrapped.render_mode = None 

for episode in range(total_episodes):
    state, _ = env.reset()
    done = False
    
    for step in range(99):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)

        # Q-Learning Formula
        qtable[state, action] = qtable[state, action] + learning_rate * (
            reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
        )
        
        state = new_state
        if terminated or truncated:
            break
            
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("✅ Training Complete! Opening window to watch the robot...")
time.sleep(1)

# --- 3. LIVE PLAYBACK ---
# Re-enable "human" mode for the visual test
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

for episode in range(3): # Watch it play 3 times
    state, _ = env.reset()
    done = False
    print(f"🎬 Episode {episode + 1}")

    while not done:
        env.render() # This opens/updates the popup window
        
        action = np.argmax(qtable[state, :])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        time.sleep(0.3) # Slows down the robot so you can watch it

env.close()