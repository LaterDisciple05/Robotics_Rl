import gymnasium as gym
import ale_py
from stable_baselines3 import PPO

# 1. Register Atari
gym.register_envs(ale_py)

# 2. Setup Env
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

# 3. Create Model
model = PPO("CnnPolicy", env, verbose=1)

# 4. Train (This creates the 'brain')
print("Training for a short time to create the file...")
model.learn(total_timesteps=10000)

# 5. Save (This creates ppo_space_invaders.zip)
model.save("ppo_space_invaders")
print("Success! 'ppo_space_invaders.zip' has been created.")