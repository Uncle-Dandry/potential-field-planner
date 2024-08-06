import numpy as np

class LocalMinimumHandler:
    def __init__(self, locmin_averaging_interval=20):
        self.locmin_averaging_interval = locmin_averaging_interval
    
    def handle_local_minimum(self, robot_trajectory, local_min_flag, local_min_obstacles, local_min_obstacles_radius, velocity, time_step):        
        if not bool(local_min_flag):
            local_min_obstacles = np.array([])
            local_min_obstacles_radius = np.array([])
        else:
            # Number of points in the trajectory. The number of points is sufficient for averaging
            i0 = robot_trajectory.shape[0]
            sigma_mean_pr = self.locmin_averaging_interval * 0.1 * velocity * time_step

            if i0 > 2 * self.locmin_averaging_interval and (i0 // self.locmin_averaging_interval) == (i0 / self.locmin_averaging_interval):
                # Get average values
                xmean1 = np.sum(robot_trajectory[i0 - 2 * self.locmin_averaging_interval:i0 - self.locmin_averaging_interval, 0]) / (self.locmin_averaging_interval + 1)
                ymean1 = np.sum(robot_trajectory[i0 - 2 * self.locmin_averaging_interval:i0 - self.locmin_averaging_interval, 1]) / (self.locmin_averaging_interval + 1)
                zmean1 = np.sum(robot_trajectory[i0 - 2 * self.locmin_averaging_interval:i0 - self.locmin_averaging_interval, 2]) / (self.locmin_averaging_interval + 1)

                xmean2 = np.sum(robot_trajectory[i0 - self.locmin_averaging_interval:i0, 0]) / self.locmin_averaging_interval
                ymean2 = np.sum(robot_trajectory[i0 - self.locmin_averaging_interval:i0, 1]) / self.locmin_averaging_interval
                zmean2 = np.sum(robot_trajectory[i0 - self.locmin_averaging_interval:i0, 2]) / self.locmin_averaging_interval
                
                if float(np.linalg.norm([xmean1 - xmean2, ymean1 - ymean2, zmean1 - zmean2])) <= sigma_mean_pr:
                    new_local_min = 0.5 * np.array([(xmean1 + xmean2), (ymean1 + ymean2), (zmean1 + zmean2)])
                    local_min_obstacles = np.hstack((local_min_obstacles, new_local_min)) if len(local_min_obstacles) == 0 else np.vstack((local_min_obstacles, new_local_min))
                    local_min_obstacles_radius = np.append(local_min_obstacles_radius, 0.8)

                    print('A local minimum was found at the point:', new_local_min)

        return local_min_obstacles, local_min_obstacles_radius
    