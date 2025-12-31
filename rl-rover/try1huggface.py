import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# Configuration
env_id = "LunarLander-v3"
model_name = "ppo_lunar_lander_model"

def main():
    # Phase 1: Training
    if not os.path.exists(model_name + ".zip"):
        print("Starting Training (No GUI mode)...")
        # We use 16 environments to maximize CPU usage for speed
        train_env = make_vec_env(env_id, n_envs=16)
        
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            verbose=1
        )
        
        # Training for 500,000 steps to ensure a high score
        model.learn(total_timesteps=500000)
        model.save(model_name)
        train_env.close()
        print("Training complete. Model saved.")
    else:
        print("Model found. Proceeding to console evaluation...")

    # Phase 2: Terminal Evaluation
    # We use render_mode=None to keep it strictly in the CLI
    eval_env = gym.make(env_id, render_mode=None)
    model = PPO.load(model_name)
    
    print("\nEvaluating Agent Performance (5 Episodes):")
    print("-" * 40)
    
    for episode in range(5):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f}")
    
    eval_env.close()
    print("-" * 40)
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()