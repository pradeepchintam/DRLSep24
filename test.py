from stable_baselines3 import PPO
from robot_env import RobotEnv
import matplotlib.pyplot as plt

# Load the trained model
model = PPO.load("ppo_robot_navigation")

# Initialize the environment with the same number of obstacles
env = RobotEnv(num_obstacles=3)

# Function to test the model
def test_model(env, model, episodes=1):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

            # Plot the robot, obstacles, start, and target locations
            plt.clf()
            plt.plot(env.robot_pos[0], env.robot_pos[1], 'bo', label='Robot')
            plt.plot(env.target_pos[0], env.target_pos[1], 'go', label='Target')
            plt.plot(env.obstacle_positions[:, 0], env.obstacle_positions[:, 1], 'ro', label='Obstacles')
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.legend()
            plt.pause(0.1)

        rewards.append(total_reward)
        print(f"Episode {ep + 1}: Total Reward = {total_reward}")
    return rewards

# Test the model
test_rewards = test_model(env, model)

# Plotting the rewards
plt.plot(test_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Test Performance of PPO Model')
plt.show()
