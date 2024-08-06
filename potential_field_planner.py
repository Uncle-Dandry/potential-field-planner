import numpy as np

from robot_trajectory import UpdateRobotTrajectory
from local_minimum import LocalMinimumHandler
from polyhedral_obstacles import create_main_obstacle, translate_polyhedrons_to_obs

class PotentialFieldPlanner:
    def __init__(
        self, num_robots, max_x, max_y, max_z_height, dt_proc=1, cruising_speed=1.5,
        follow_leader=False, maintain_distance=False
    ):
        self.num_robots = num_robots
        self.max_x = max_x
        self.max_y = max_y
        self.max_z_height = max_z_height
        self.dt_proc = dt_proc
        self.cruising_speed = cruising_speed
        self.follow_leader = follow_leader
        self.maintain_distance = maintain_distance
        self.obstacle_radius_multiplier = 1.1
        self.locmin_averaging_interval = 25
        self.Rmaxgr = 4
        self.inner_robot_radius = 0.5
        self.goal_radius = 2 * self.inner_robot_radius

    def plan_group_trajectories(self, start_points, end_points, obstacle_heights, vertices, faces, field_sizes):
        # Initialize the motion trajectory array
        routes = [[] for _ in range(self.num_robots)]
        target_reached_flags = [False] * self.num_robots
        
        # Initialization of arrays for detecting local minima
        obstacles_locmin = [[] for _ in range(self.num_robots)]
        obstacles_radius_locmin = [[] for _ in range(self.num_robots)]
        increment_locmin = [0] * self.num_robots
        flag_locmin = [1] * self.num_robots
        
        # Initialization of arrays of obstacles generated for a given UAV by other UAVs
        obstacles_inner = np.zeros((self.num_robots, 3))
        obstacles_radius_inner = np.full(self.num_robots, self.inner_robot_radius)
        
        # Filling obstacles with starting points
        for robot_number in range(self.num_robots):
            routes[robot_number] = start_points[:, robot_number]
            obstacles_inner[robot_number] = routes[robot_number]
        
        # Assignment of a cube representing the underlying surface
        obstacle_radius_main, obstacles_main = create_main_obstacle(self.max_x, self.max_y, self.obstacle_radius_multiplier)
        
        max_points, current_point, flag_out = 500, 1, False
        
        update_robot_trajectory = UpdateRobotTrajectory()
        local_minimum_handler = LocalMinimumHandler(self.locmin_averaging_interval)
        
        while not flag_out and current_point <= max_points:
            for robot_index in range(self.num_robots):
                if target_reached_flags[robot_index]:
                    continue
                
                if (current_point == 1):
                    obstacles_locmin[robot_index] = []
                    obstacles_radius_locmin[robot_index] = []
                    
                increment_locmin[robot_index] += 1

                if increment_locmin[robot_index] >= 4 * self.locmin_averaging_interval:
                    increment_locmin[robot_index] = 0
                    flag_locmin = [abs(1 - flag) for flag in flag_locmin]

                current_robot_point = routes[robot_index] if np.array(routes[robot_index]).ndim == 1 else routes[robot_index][-1]
                obstacles, obstacles_radius, distances_to_obstacles = translate_polyhedrons_to_obs(current_robot_point, faces, vertices,
                    self.cruising_speed, self.dt_proc)
                    
                # receiving the positions of current obstacles - other UAVs in the group
                obstacles_inner_current_robot = np.delete(obstacles_inner, robot_index, axis=0)
                obstacles_radius_inner_current_robot = np.delete(obstacles_radius_inner, robot_index)
                
                # If the maintain_distance is active, check the maximum permissible distance of the slave UAV from the master
                if self.maintain_distance:
                    maintain_robot_point = np.array(routes[0][-1])
                    maintain_robot_distance = np.linalg.norm(current_robot_point - maintain_robot_point)

                    if robot_index != 0 and maintain_robot_distance > 0.9 * self.Rmaxgr:
                        # create a virtual obstacle
                        maintain_robot_point_radius = 0.1 * maintain_robot_distance
                        direction_vector = (current_robot_point - maintain_robot_point) / maintain_robot_distance
                        maintain_virtual_obstacle = maintain_robot_point + (maintain_robot_distance + 2 * maintain_robot_point_radius) * direction_vector
                        obstacles_inner_current_robot = np.vstack((obstacles_inner_current_robot, maintain_virtual_obstacle))
                        obstacles_radius_inner_current_robot = np.append(obstacles_radius_inner_current_robot, maintain_robot_point_radius)
                
                if self.follow_leader:
                    if robot_index == 0:
                        goal_radius_h = self.goal_radius
                        end_point = end_points[:, robot_index]
                    else:
                        goal_radius_h = 0.4
                        end_point = routes[0][-1]
                    
                    if target_reached_flags[0]:
                        end_point = end_points[:, robot_index]
                        goal_radius_h = self.goal_radius
                else:
                    end_point = end_points[:, robot_index]
                    goal_radius_h = self.goal_radius
                    
                # Combining external and internal obstacles
                combined_obstacles = np.vstack((obstacles_main, obstacles_inner_current_robot))
                combined_obstacles_radius = np.append(obstacle_radius_main, obstacles_radius_inner_current_robot)

                obstacles_locmin_new, obstacles_radius_locmin_new = local_minimum_handler.handle_local_minimum(
                    routes[robot_index], flag_locmin, obstacles_locmin[robot_index],
                    obstacles_radius_locmin[robot_index], self.cruising_speed, self.dt_proc)

                if len(obstacles_locmin_new) != 0:
                    combined_obstacles_radius = np.append(combined_obstacles_radius, obstacles_radius_locmin_new)
                    combined_obstacles = np.vstack((combined_obstacles, obstacles_locmin_new))

                obstacles_locmin[robot_index] = obstacles_locmin_new
                obstacles_radius_locmin[robot_index] = obstacles_radius_locmin_new
                
                # Call and run the main solver for the one-step planner
                robot_trajectory_new, target_reached_flag_new = update_robot_trajectory.get_new_robot_trajectory(
                    routes[robot_index], current_robot_point, end_point, combined_obstacles, combined_obstacles_radius,
                    obstacles, obstacles_radius, distances_to_obstacles, self.dt_proc, self.cruising_speed,
                    goal_radius_h)

                routes[robot_index] = robot_trajectory_new
                obstacles_inner[robot_index] = robot_trajectory_new[-1]

                target_reached_flags[robot_index] = target_reached_flag_new
                flag_out = all(target_reached_flags)
                
            current_point += 1

        return routes, target_reached_flags
    