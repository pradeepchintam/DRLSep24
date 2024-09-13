from stable_baselines3 import PPO
from robot_env import RobotEnv
import matplotlib.pyplot as plt
import numpy as np
import time
import advised_rrt_env as arrtenv

# Load the trained model
model = PPO.load("ppo_robot_navigation")
envToUse = arrtenv.EnvA()

# Define the same static rectangular obstacles as during training
static_obstacles = arrtenv.EnvA.obs_rectangle()

# Initialize the environment for testing
env = RobotEnv(num_dynamic_obstacles=3, static_obstacles=static_obstacles, robot_pos=envToUse.s_start, target_pos=envToUse.s_goal)

# Function to draw static obstacles on the plot
def draw_static_obstacles(static_obstacles):
    for (x, y, width, height) in static_obstacles:
        rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
        plt.gca().add_patch(rect)

# Function to plot the robot, dynamic obstacles, and target
def plot_env(robot_pos, dynamic_obstacles, target_pos, static_obstacles):
    plt.clf()  # Clear the current figure

    # Plot static obstacles
    draw_static_obstacles(static_obstacles)

    # Plot dynamic obstacles
    for obstacle in dynamic_obstacles:
        plt.scatter(obstacle[0], obstacle[1], color='blue', s=100, label="Dynamic Obstacle" if obstacle is dynamic_obstacles[0] else "")

    # Plot robot position
    plt.scatter(robot_pos[0], robot_pos[1], color='green', s=100, marker='o', label="Robot")

    # Plot target position
    plt.scatter(target_pos[0], target_pos[1], color='red', s=100, marker='x', label="Target")

    # Set plot limits
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # Add legend
    plt.legend(loc='upper left')

    # Add grid for better visualization
    plt.grid(True)

    # Draw the updated plot
    plt.pause(0.1)

# Function to test the model and visualize the navigation
def test_model_with_visualization(env, model, episodes=1):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        plt.figure(figsize=(6, 6))
        plt.ion()  # Turn on interactive mode for live plotting

        # Loop through each step until the episode is done
        while not done:
            # Get the action from the model
            action, _ = model.predict(obs, deterministic=True)

            # Take a step in the environment
            obs, reward, done, _, info = env.step(action)

            # Extract positions from the observation (robot and dynamic obstacles)
            robot_pos = obs[:2]
            dynamic_obstacles = obs[2:].reshape(-1, 2)

            # Plot the current state of the environment
            plot_env(robot_pos, dynamic_obstacles, env.target_pos, env.static_obstacles)

            # Add a small delay to simulate real-time movement
            time.sleep(0.1)

        # Close the plot after the episode is finished
        plt.ioff()
        plt.show()

# Test the model with visualization
test_model_with_visualization(env, model)
