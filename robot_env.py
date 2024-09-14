import gymnasium as gym
import numpy as np


class RobotEnv(gym.Env):
    def __init__(self, num_dynamic_obstacles=1, static_obstacles=None, robot_pos=None, target_pos=None, training = False):
        super(RobotEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # [0: left, 1: right, 2: up, 3: down]

        # Observation space (robot position + dynamic obstacles positions)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(2 + 2 * num_dynamic_obstacles,), dtype=np.float32)

        # Initialize robot, target, and dynamic obstacles
        self.robot_pos = np.array(robot_pos if robot_pos is not None else [0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array(target_pos if target_pos is not None else [10.0, 10.0], dtype=np.float32)

        # Dynamic obstacles: randomly placed at the start and move randomly
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.dynamic_obstacles = np.random.uniform(0, 10, (num_dynamic_obstacles, 2)).astype(np.float32)

        self.static_obstacles = static_obstacles

        self.time_step = 0
        self.max_time_steps = 100
        self.training = training

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array(self.robot_pos, dtype=np.float32)
        self.target_pos = np.array(self.target_pos, dtype=np.float32)
        self.dynamic_obstacles = np.random.uniform(0, 10, (self.num_dynamic_obstacles, 2)).astype(np.float32)
        self.time_step = 0

        # Return the robot position and dynamic obstacle positions
        return np.concatenate([self.robot_pos, self.dynamic_obstacles.flatten()]).astype(np.float32), {}

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

        # Clip the robot position to stay within bounds [0, 10]
        self.robot_pos = np.clip(self.robot_pos, 0, 10)

        # Update dynamic obstacles positions
        self.dynamic_obstacles += np.random.uniform(-0.5, 0.5, (self.num_dynamic_obstacles, 2)).astype(np.float32)
        self.dynamic_obstacles = np.clip(self.dynamic_obstacles, 0, 10)

        # Calculate distances to target and obstacles
        distance_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        distance_to_dynamic_obstacles = np.linalg.norm(self.robot_pos - self.dynamic_obstacles, axis=1)

        # Reward: negative distance to target (closer is better)
        reward = -float(distance_to_target)

        # Penalize for getting too close to dynamic obstacles
        if np.any(distance_to_dynamic_obstacles < 1):
            reward -= 10.0

        # Penalize for hitting any static obstacle
        if self._check_static_obstacle_collision():
            reward -= 10.0

        # Check if the episode is done (target reached or max steps)
        done = distance_to_target < 1 or self.time_step >= self.max_time_steps
        if (distance_to_target < 0.5) & self.training:
            # Update target position by a random value between 1 and 5

            self.target_pos += np.random.uniform(-5, 5, 2).astype(np.float32)
            self.target_pos = np.clip(self.target_pos, 0, 10)
            print('Changing target position -------- ', self.target_pos)

        # Return state (robot and dynamic obstacle positions)
        state = np.concatenate([self.robot_pos, self.dynamic_obstacles.flatten()]).astype(np.float32)
        return state, reward, done, False, {}

    def _check_static_obstacle_collision(self):
        """
        Check if the robot collides with any static rectangular obstacles.
        Static obstacles are defined by their bottom-left corner, width, and height.
        """
        for (x, y, width, height) in self.static_obstacles:
            if x <= self.robot_pos[0] <= x + width and y <= self.robot_pos[1] <= y + height:
                return True  # Collision detected
        return False

    def render(self, mode="human"):
        # Optional rendering logic (could visualize the environment using matplotlib or Pygame)
        pass