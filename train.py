from stable_baselines3 import PPO
from robot_env import RobotEnv

# Define some static rectangular obstacles
static_obstacles = [
    (3.0, 3.0, 2.0, 2.0),  # Static rectangular obstacle
    (7.0, 7.0, 1.0, 3.0)   # Static rectangular obstacle
]

# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_robot_navigation")

print("Training complete. Model saved as 'ppo_robot_navigation'.")
