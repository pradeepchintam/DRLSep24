from stable_baselines3 import PPO
from robot_env import RobotEnv

# Initialize environment with multiple obstacles (e.g., 3 obstacles)
env = RobotEnv(num_obstacles=3)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_robot_navigation")

print("Training complete. Model saved as 'ppo_robot_navigation'.")
