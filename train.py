from stable_baselines3 import PPO
from robot_env import RobotEnv
import advised_rrt_env as arrtenv

envToUse = arrtenv.EnvA()
static_obstacles = arrtenv.EnvA.obs_rectangle()

# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=(9,9), target_pos=(1,1), training=True)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=40000)

# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=(1, 1), target_pos=(9, 9), training=True)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=40000)



# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=(2,8), target_pos=(8,2), training=True)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=40000)

# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=(3,9), target_pos=(9,3), training=True)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=40000)

# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=(0,0), target_pos=(10,10), training=True)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=40000)

# Save the trained model
model.save("ppo_robot_navigation")

print("Training complete. Model saved as 'ppo_robot_navigation'.")
