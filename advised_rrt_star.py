import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class AdvisedRRTStar:
    def __init__(self, start, goal, obstacle_list, search_radius=2.0, max_iterations=10000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]
        self.search_radius = search_radius
        self.max_iterations = max_iterations

    def distance(self, node1, node2):
        return np.linalg.norm([node1.x - node2.x, node1.y - node2.y])

    def is_collision_free(self, node1, node2):
        # Check if the path from node1 to node2 collides with any obstacles
        for (ox, oy, width, height) in self.obstacle_list:
            if self.check_line_rect_collision(node1.x, node1.y, node2.x, node2.y, ox, oy, width, height):
                return False
        return True

    def check_line_rect_collision(self, x1, y1, x2, y2, rx, ry, rw, rh):
        # Check if a line from (x1, y1) to (x2, y2) intersects with a rectangle
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh:
            return False
        return True

    def steer(self, from_node, to_node, extend_length=float("inf")):
        dist = self.distance(from_node, to_node)
        if dist <= extend_length:
            return Node(to_node.x, to_node.y)
        else:
            angle = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node = Node(from_node.x + extend_length * math.cos(angle),
                            from_node.y + extend_length * math.sin(angle))
            new_node.parent = from_node
            return new_node

    def random_sampling(self):
        return Node(random.uniform(0, 10), random.uniform(0, 10))

    def advised_sampling(self):
        c_min = self.distance(self.start, self.goal)
        x_center = np.array([(self.start.x + self.goal.x) / 2.0, (self.start.y + self.goal.y) / 2.0])
        L = np.diag([c_min / 2.0, np.sqrt(c_min ** 2 / 2.0)])

        while True:
            q = np.random.uniform(-1, 1, 2)
            q /= np.linalg.norm(q)
            sample = L @ q + x_center
            if 0 <= sample[0] <= 10 and 0 <= sample[1] <= 10:
                return Node(sample[0], sample[1])

    def get_nearest_node(self, random_node):
        return min(self.node_list, key=lambda node: self.distance(node, random_node))

    def get_near_nodes(self, new_node):
        n = len(self.node_list)
        r = min(self.search_radius, np.sqrt(np.log(n) / n))  # Adaptive search radius
        return [node for node in self.node_list if self.distance(node, new_node) <= r]

    def get_final_path(self, goal_node):
        path = []
        path.append([self.goal.x, self.goal.y])
        node = goal_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path[::-1]

    def plan(self):
        plt.ion()
        fig, ax = plt.subplots()
        for (ox, oy, width, height) in self.obstacle_list:
            ax.add_patch(plt.Rectangle((ox, oy), width, height, edgecolor='r', facecolor='r', alpha=0.5))
        ax.scatter(self.start.x, self.start.y, c='b', label='Start')
        ax.scatter(self.goal.x, self.goal.y, c='r', label='Goal')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.grid(True)
        ax.legend()
        final_path = None

        for _ in range(self.max_iterations):
            # Advised sampling
            if final_path:
                random_node = self.advised_sampling()
            else:
                random_node = self.random_sampling()
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node, extend_length=0.5)

            # If the new node is collision-free, add to the tree
            if self.is_collision_free(nearest_node, new_node):
                near_nodes = self.get_near_nodes(new_node)
                new_node.parent = nearest_node  # Set the parent
                self.node_list.append(new_node)

                # Rewire the tree to optimize the path
                for near_node in near_nodes:
                    if self.is_collision_free(new_node, near_node) and \
                            self.distance(new_node, near_node) < self.distance(near_node, near_node.parent)+1:
                        # Ensure no cycle is created
                        if near_node != new_node.parent:
                            near_node.parent = new_node

                ax.plot([nearest_node.x, new_node.x], [nearest_node.y, new_node.y], 'g-')
                plt.pause(0.01)

            # Check if we reached the goal
            if self.distance(new_node, self.goal) <= 1.0:
                print("Goal reached!")
                final_path = self.get_final_path(new_node)
                ax.plot([node[0] for node in final_path], [node[1] for node in final_path], 'b-', linewidth=2)
                plt.pause(0.01)

        plt.ioff()
        plt.show()
        # Return empty if no path found
        return final_path, len(self.node_list), self.compute_path_length(final_path)

    def compute_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        return length

# Example usage:
if __name__ == "__main__":
    start_pos = [0, 0]
    goal_pos = [9, 9]

    # Static obstacles: List of (x, y, width, height) for rectangular obstacles
    static_obstacles = [
        (3.0, 3.0, 2.0, 2.0),  # A 2x2 rectangle at position (3, 3)
        (7.0, 7.0, 1.0, 3.0)   # A 1x3 rectangle at position (7, 7)
    ]

    rrt_star = AdvisedRRTStar(start=start_pos, goal=goal_pos, obstacle_list=static_obstacles)
    path, num_nodes, path_length = rrt_star.plan()

    if path:
        print(f"Path found with length: {path_length}")
        print(f"Number of nodes used: {num_nodes}")
        print(f"Path: {path}")

        # Plot the path
        plt.figure()
        for (ox, oy, width, height) in static_obstacles:
            plt.gca().add_patch(plt.Rectangle((ox, oy), width, height, edgecolor='r', facecolor='r', alpha=0.5))

        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'g', label='Path')
        plt.scatter(path[:, 0], path[:, 1], c='g')

        plt.scatter(start_pos[0], start_pos[1], c='b', label='Start')
        plt.scatter(goal_pos[0], goal_pos[1], c='r', label='Goal')

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("No path found!")

