import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
import time

# 1. Register Atari games so Gymnasium can find them
gym.register_envs(ale_py)

# 2. Create the environment in visual mode
# Use "ALE/SpaceInvaders-v5" for the newest version
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# 3. Load your best model
# Ensure the .zip file exists in your folder!
MODEL_NAME = "ppo_space_invaders" 

try:
    model = PPO.load(MODEL_NAME)
    print(f"--- Model '{MODEL_NAME}' loaded successfully! ---")
except Exception as e:
    print(f"Error: Could not find {MODEL_NAME}.zip. Did you finish training?")
    exit()

# 4. The Game Loop
obs, info = env.reset()
print("Starting game... Close the window or press Ctrl+C in terminal to stop.")

try:
    while True:
        # Agent chooses an action
        action, _states = model.predict(obs, deterministic=True)
        
        # Agent takes the action in the game
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Slow down the loop slightly so it looks natural to humans
        time.sleep(0.01)
        
        # Reset if the game ends (game over)
        if terminated or truncated:
            print("Game Over! Resetting...")
            obs, info = env.reset()
            
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    env.close()