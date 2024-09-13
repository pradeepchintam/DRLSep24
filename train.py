from stable_baselines3 import PPO
from robot_env import RobotEnv
import advised_rrt_env as arrtenv

envToUse = arrtenv.EnvA()
static_obstacles = arrtenv.EnvA.obs_rectangle()

# Initialize the environment with dynamic and static obstacles
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=envToUse.s_start, target_pos=envToUse.s_goal)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_robot_navigation")

print("Training complete. Model saved as 'ppo_robot_navigation'.")
