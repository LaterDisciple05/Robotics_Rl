import gymnasium as gym
import ale_py
from stable_baselines3 import PPO

gym.register_envs(ale_py)

# Load the environment in human mode
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# Load the 'Master' model
model = PPO.load("space_invaders_master")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()