import gymnasium as gym
import numpy as np

class RobotEnv(gym.Env):
    def __init__(self, num_obstacles=1):
        super(RobotEnv, self).__init__()
        # Define action space (e.g., move in four directions)
        self.action_space = gym.spaces.Discrete(4)  # [0: left, 1: right, 2: up, 3: down]

        # Define observation space (robot position and multiple obstacles)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(2 + 2 * num_obstacles,), dtype=np.float32)

        # Initialize state (robot's initial position, target, and obstacles)
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([10.0, 10.0], dtype=np.float32)

        self.num_obstacles = num_obstacles
        self.obstacle_positions = np.random.uniform(0, 10, (num_obstacles, 2)).astype(np.float32)

        self.time_step = 0
        self.max_time_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([10.0, 10.0], dtype=np.float32)
        self.obstacle_positions = np.random.uniform(0, 10, (self.num_obstacles, 2)).astype(np.float32)
        self.time_step = 0

        # Combine robot and obstacle positions for observation
        return np.concatenate([self.robot_pos, self.obstacle_positions.flatten()]).astype(np.float32), {}

    def step(self, action):
        self.time_step += 1
        # Move the robot based on the action
        if action == 0:  # move left
            self.robot_pos[0] -= 1
        elif action == 1:  # move right
            self.robot_pos[0] += 1
        elif action == 2:  # move up
            self.robot_pos[1] += 1
        elif action == 3:  # move down
            self.robot_pos[1] -= 1

        # Clip the robot position to be within the bounds [0, 10]
        self.robot_pos = np.clip(self.robot_pos, 0, 10)

        # Update dynamic obstacle positions (move randomly) and clip them
        self.obstacle_positions += np.random.uniform(-0.5, 0.5, (self.num_obstacles, 2)).astype(np.float32)
        self.obstacle_positions = np.clip(self.obstacle_positions, 0, 10)

        # Compute distances to target and obstacles
        distance_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        distance_to_obstacles = np.linalg.norm(self.robot_pos - self.obstacle_positions, axis=1)

        # Reward: negative distance to the target
        reward = -float(distance_to_target)

        # Penalize for getting too close to any obstacle
        if np.any(distance_to_obstacles < 1):
            reward -= 10.0

        # Check if the episode is over (target reached or max time steps)
        done = distance_to_target < 1 or self.time_step >= self.max_time_steps

        # Return state (robot position and obstacle positions)
        state = np.concatenate([self.robot_pos, self.obstacle_positions.flatten()]).astype(np.float32)
        return state, reward, done, False, {}

    def render(self, mode="human"):
        # Optional rendering logic
        pass
