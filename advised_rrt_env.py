class EnvX:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (18, 8) # Starting node
        self.s_goal = (37, 18) # Goal node
        self.step_len = 0.8
        self.goal_sample_rate = 0.05
        self.sampleRadius = 10
        self.iter_max = 1000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

        return obs_cir

class EnvA:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (12, 12) # Starting node
        self.s_goal = (28, 28) # Goal node
        self.step_len = 4
        self.goal_sample_rate = 0.05
        self.search_radius = 10
        self.iter_max = 2000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [16, 12, 8, 16]
        ]

        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [0, 0, 0]
        ]

        return obs_cir

class EnvB:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (2, 2)  # Starting node
        self.s_goal = (38, 38)  # Goal node
        self.step_len = 4
        self.goal_sample_rate = 0.05
        self.search_radius = 10
        self.iter_max = 500
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [0, 18, 4, 8],
            [2, 32, 6, 6],
            [10, 10, 6, 10],
            [16, 26, 12, 6],
            [28, 4, 10, 14]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir

class EnvC:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (1.2, 2.5)  # Starting node
        self.s_goal = (38.5, 38.5)  # Goal node
        self.step_len = 4
        self.goal_sample_rate = 0.05
        self.search_radius = 10
        self.iter_max = 3000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [0.8,0.6,1.8,1.2],
            [1.6,2.52,1.8,1.2],
            [0.64,4.8,1.8,1.2],
            [1.04,6.48,1.8,1.2],
            [1.04,8.52,1.8,1.2],
            [0.4,10.64,1.8,1.2],
            [1.04,12.4,1.8,1.2],
            [0.6,14.68,1.8,1.2],
            [0.76,16.72,1.8,1.2],
            [1.24,18.6,1.8,1.2],
            [1.16,20.72,1.8,1.2],
            [1.6,22.48,1.8,1.2],
            [1.32,24.56,1.8,1.2],
            [1.24,26.52,1.8,1.2],
            [0.8,28.8,1.8,1.2],
            [1,30.52,1.8,1.2],
            [0.72,32.72,1.8,1.2],
            [0.64,34.8,1.8,1.2],
            [4.28,2.68,1.2,2.8],
            [4.12,6.8,1.2,2.8],
            [4.8,10.76,1.2,2.8],
            [4.8,14.48,1.2,2.8],
            [3.4,18.72,1.2,2.8],
            [3.4,22.64,1.2,2.8],
            [3.64,26.6,1.2,2.8],
            [4.28,30.64,1.2,2.8],
            [5,34.56,1.2,2.8],
            [7,0.76,1.8,1.2],
            [6.56,2.48,1.8,1.2],
            [6.72,4.64,1.8,1.2],
            [6.72,6.52,1.8,1.2],
            [6.48,8.72,1.8,1.2],
            [6.64,10.48,1.8,1.2],
            [7.44,12.76,1.8,1.2],
            [7.52,14.4,1.8,1.2],
            [7.2,16.76,1.8,1.2],
            [6.48,18.56,1.8,1.2],
            [7.04,20.48,1.8,1.2],
            [6.52,22,1.8,1.2],
            [7.12,24.4,1.8,1.2],
            [7.12,26.8,1.8,1.2],
            [7.16,28.52,1.8,1.2],
            [6.6,30.8,1.8,1.2],
            [6.88,32,1.8,1.2],
            [6.4,34.52,1.8,1.2],
            [6.48,36.52,1.8,1.2],
            [9.64,2.72,1.2,2.8],
            [10.16,6.72,1.2,2.8],
            [10.48,10.44,1.2,2.8],
            [10.64,14.4,1.2,2.8],
            [9.56,18.56,1.2,2.8],
            [9.52,22.52,1.2,2.8],
            [9.68,26.68,1.2,2.8],
            [9.8,30.76,1.2,2.8],
            [10.68,34.4,1.2,2.8],
            [12.88,0.68,1.8,1.2],
            [13.56,2.72,1.8,1.2],
            [13.56,4.8,1.8,1.2],
            [13.2,6.72,1.8,1.2],
            [12.64,8.8,1.8,1.2],
            [12.48,10.48,1.8,1.2],
            [12.64,12.8,1.8,1.2],
            [12.8,14.52,1.8,1.2],
            [12.48,16.72,1.8,1.2],
            [12.88,18.72,1.8,1.2],
            [12.64,20.68,1.8,1.2],
            [13.56,22.68,1.8,1.2],
            [13.44,24.72,1.8,1.2],
            [13.48,26.4,1.8,1.2],
            [12.76,28.52,1.8,1.2],
            [13.44,30.72,1.8,1.2],
            [13.24,32.6,1.8,1.2],
            [12.44,34.52,1.8,1.2],
            [13.16,36.72,1.8,1.2],
            [16.24,2.6,1.2,2.8],
            [16.92,6.44,1.2,2.8],
            [15.88,10.64,1.2,2.8],
            [16.32,14.76,1.2,2.8],
            [16.36,18.52,1.2,2.8],
            [16.6,22.44,1.2,2.8],
            [16,26.72,1.2,2.8],
            [16.44,30.68,1.2,2.8],
            [15.92,34.44,1.2,2.8],
            [18.64,0.8,1.8,1.2],
            [19.08,2.48,1.8,1.2],
            [18.48,4.4,1.8,1.2],
            [18.44,6.72,1.8,1.2],
            [19.32,8.64,1.8,1.2],
            [18.8,10.6,1.8,1.2],
            [19.48,12.56,1.8,1.2],
            [19.4,14.52,1.8,1.2],
            [18.84,16.48,1.8,1.2],
            [18.44,18.8,1.8,1.2],
            [18.92,20.72,1.8,1.2],
            [18.4,22.52,1.8,1.2],
            [19.4,24.4,1.8,1.2],
            [18.88,26.8,1.8,1.2],
            [18.88,28.8,1.8,1.2],
            [19.4,30.44,1.8,1.2],
            [19.48,32.68,1.8,1.2],
            [18.72,34.52,1.8,1.2],
            [18.84,36.56,1.8,1.2],
            [21.76,2.6,1.2,2.8],
            [21.64,6.4,1.2,2.8],
            [21.28,10.64,1.2,2.8],
            [21.84,14.56,1.2,2.8],
            [21.52,18.52,1.2,2.8],
            [22.6,22.6,1.2,2.8],
            [21.64,26.6,1.2,2.8],
            [21.6,30.44,1.2,2.8],
            [21.44,34.52,1.2,2.8],
            [25.48,0.8,1.8,1.2],
            [24.64,2.8,1.8,1.2],
            [25.2,4.8,1.8,1.2],
            [25.2,6.8,1.8,1.2],
            [25.08,8.56,1.8,1.2],
            [24.8,10.48,1.8,1.2],
            [25.2,12.68,1.8,1.2],
            [25.36,14.52,1.8,1.2],
            [24.6,16.6,1.8,1.2],
            [24.84,18.72,1.8,1.2],
            [25.56,20.64,1.8,1.2],
            [24.64,22.68,1.8,1.2],
            [24.44,24.6,1.8,1.2],
            [25.36,26.4,1.8,1.2],
            [25.44,28.52,1.8,1.2],
            [24.6,30.48,1.8,1.2],
            [24.68,32.72,1.8,1.2],
            [24.68,34.6,1.8,1.2],
            [25,36.6,1.8,1.2],
            [28.48,2.48,1.2,2.8],
            [28.64,6.64,1.2,2.8],
            [27.44,10.4,1.2,2.8],
            [28.4,14.4,1.2,2.8],
            [27.48,18.4,1.2,2.8],
            [28.08,22.68,1.2,2.8],
            [27.48,26.72,1.2,2.8],
            [28.24,30.56,1.2,2.8],
            [28.28,34.8,1.2,2.8],
            [31.28,0.8,1.8,1.2],
            [31.24,2.52,1.8,1.2],
            [31.52,4.48,1.8,1.2],
            [31.4,6.52,1.8,1.2],
            [30.76,8.48,1.8,1.2],
            [30.44,10.4,1.8,1.2],
            [31.24,12.76,1.8,1.2],
            [30.92,14.76,1.8,1.2],
            [30.8,16.44,1.8,1.2],
            [30.72,18.76,1.8,1.2],
            [30.52,20.48,1.8,1.2],
            [31.16,22.44,1.8,1.2],
            [31.56,24.52,1.8,1.2],
            [31.56,26.64,1.8,1.2],
            [31.04,28.4,1.8,1.2],
            [31.44,30.6,1.8,1.2],
            [31.16,32.8,1.8,1.2],
            [30.52,34.48,1.8,1.2],
            [30.6,36.72,1.8,1.2],
            [34.52,1.8,1.2,2.8],
            [35,6.48,1.2,2.8],
            [33.68,10.52,1.2,2.8],
            [33.96,14.56,1.2,2.8],
            [34.76,18.64,1.2,2.8],
            [34.24,22.56,1.2,2.8],
            [33.52,26.4,1.2,2.8],
            [34.12,30.76,1.2,2.8],
            [34.36,34.4,1.2,2.8],
            [37.6,0.6,1.8,1.2],
            [36.84,2.56,1.8,1.2],
            [36.56,4.64,1.8,1.2],
            [37.32,6.52,1.8,1.2],
            [36.44,8.52,1.8,1.2],
            [36.4,10.76,1.8,1.2],
            [37.32,12.76,1.8,1.2],
            [36.84,14.68,1.8,1.2],
            [37,16.6,1.8,1.2],
            [36.88,18.72,1.8,1.2],
            [37.08,20.56,1.8,1.2],
            [36.96,22.52,1.8,1.2],
            [36.72,24.52,1.8,1.2],
            [36.96,26.8,1.8,1.2],
            [36.68,28.44,1.8,1.2],
            [36.92,30.48,1.8,1.2],
            [37,32.56,1.8,1.2],
            [37.2,34.76,1.8,1.2],
            [37.12,36.52,1.8,1.2],
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir

class EnvD:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (2, 2)  # Starting node
        self.s_goal = (38, 38)  # Goal node
        self.step_len = 2
        self.goal_sample_rate = 0.03
        self.search_radius = 10
        self.iter_max = 3000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [0,16,4,1],
            [0,32,4,1],
            [4,16,1,12],
            [4,28,4,1],
            [4,36,1,4],
            [4,36,12,1],
            [6,4,1,4],
            [6,8,12,1],
            [8,16,1,13],
            [8,32,1,4],
            [8,32,5,1],
            [12,8,1,24],
            [12,0,1,4],
            [18,4,1,5],
            [24,0,1,4],
            [24,4,12,1],
            [36,4,1,4],
            [28,8,9,1],
            [28,8,1,4],
            [32,12,8,1],
            [13,20,4,1],
            [16,16,1,4],
            [17,16,4,1],
            [20,12,1,4],
            [13,34,12,1],
            [13,34,1,3],
            [18,30,1,4],
            [12,24,10,1],
            [30,22,10,1],
            [20,24,1,6],
            [22,20,1,10],
            [22,20,2,1],
            [24,18,1,3],
            [24,17,6,1],
            [24,14,6,1],
            [29,14,1,4],
            [28,36,12,1],
            [28,28,1,8],
            [28,28,8,1],
            [36,28,1,3],
            [32,31,5,1],
            [22,37,1,3],
            [12,38,1,2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir

# Done
class EnvE:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (8, 8)  # Starting node
        self.s_goal = (18, 14)  # Goal node
        self.step_len = 2
        self.goal_sample_rate = 0.0
        self.search_radius = 10
        self.iter_max = 6000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [0, 0, 40, 2],
            [0, 0, 1, 40],
            [0, 38, 30, 2],
            [38, 0, 1, 40],
            [8, 32, 1, 8],
            [12, 0, 2, 16],
            [12, 0, 2, 16],
            [12, 24, 1, 8],
            [29, 10, 2, 30],
            [12, 16, 10, 2],
            [12, 24, 14, 2],
            [20, 16, 1, 8],
            [20, 24, 1, 8],
            [16, 32, 1, 8],
            [24, 4, 1, 32]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir

# Done
class EnvF:

    # def __init__(self):
    #     self.x_range = (0, 100)
    #     self.y_range = (0, 100)
    #     self.obs_boundary = self.obs_boundary()
    #     self.obs_circle = self.obs_circle()
    #     self.obs_rectangle = self.obs_rectangle()
    #     self.s_start = (45, 5)  # Starting node
    #     self.s_goal = (45, 85)  # Goal node
    #     self.step_len = 2
    #     self.goal_sample_rate = 0.02
    #     self.search_radius = 5
    #     self.iter_max = 2000
    #     self.eta = 2
    #
    # @staticmethod
    # def obs_boundary():
    #     obs_boundary = [
    #         [0, 0, 1, 100],
    #         [0, 100, 100, 1],
    #         [1, 0, 100, 1],
    #         [100, 1, 1, 100]
    #     ]
    #     return obs_boundary
    #
    # @staticmethod
    # def obs_rectangle():
    #     obs_rectangle = [
    #         [45, 10, 37.5, 10],
    #         [10, 25, 37.5, 10],
    #         [45, 40, 37.5, 10],
    #         [10, 55, 37.5, 10],
    #         [45, 70, 37.5, 10]
    #     ]
    #     return obs_rectangle


    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (8, 2)  # Starting node
        self.s_goal = (32, 38)  # Goal node
        self.step_len = 2
        self.goal_sample_rate = 0.01
        self.search_radius = 10
        self.iter_max = 2500
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [18.8, 4, 20.8, 4],
            [0.4, 10.8, 20.8, 4],
            [18.8, 17.6, 20.8, 4],
            [0.4, 24.4, 20.8, 4],
            [18.8, 31.2, 20.8, 4]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir

class EnvG:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (38.5, 38.5)  # Starting node
        self.s_goal = (20, 20)  # Goal node
        self.step_len = 4
        self.goal_sample_rate = 0.001
        self.search_radius = 10
        self.iter_max = 3000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [2,36,28,1],
            [2,2,1,34],
            [2,2,34,1],
            [36,2,1,26],
            [10,28, 27, 1],
            [10,10,1,18],
            [10, 10, 16, 1],
            [26,10,1,12],
            [17,22,10,1]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir
# Done
class EnvH:
    def __init__(self):
        self.x_range = (0, 40)
        self.y_range = (0, 40)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.s_start = (2, 38)  # Starting node
        self.s_goal = (38, 2)  # Goal node
        self.step_len = 2
        self.goal_sample_rate = 0.001
        self.search_radius = 10
        self.iter_max = 2000
        self.eta = 2

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 40],
            [0, 40, 40, 1],
            [1, 0, 40, 1],
            [40, 1, 1, 40]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [0,0,10,2],
            [0,10,10,2],
            [0,20,10,2],
            [0,30,10,2],
            [8,6,10,2],
            [8,18,10,2],
            [10,26,10,4],
            [12,6,2,10],
            [14,24,2,10],
            [20,14,2,10],
            [24,14,2,10],
            [30,6,10,2],
            [24,14,10,2],
            [24,22,10,2],
            [30,26,10,4],
            [26,36,10,2]
        ]

        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        return obs_cir