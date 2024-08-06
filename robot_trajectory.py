import numpy as np

class UpdateRobotTrajectory:
    def __init__(self):
        self.Rpred = 5
        self.Rpred_2 = 0.3 * self.Rpred
        self.RpredB = min([self.Rpred / 6, 15])
        self.RB = 10
        self.local_radius = 40

    def get_new_robot_trajectory(
        self, robot_trajectory, current_position, goal_point, combined_obstacles, combined_obstacles_radius,
        polyhedron_obstacles, polyhedron_obstacle_radius, polyhedron_obstacle_distance, time_step, velocity, Rgoal
    ):
        is_goal_reached = False

        delta_x = goal_point[0] - current_position[0]
        delta_y = goal_point[1] - current_position[1]
        delta_z = goal_point[2] - current_position[2]
        distance_to_goal = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        if combined_obstacles.size == 0:
            next_position = current_position + velocity * time_step * np.array([delta_x, delta_y, delta_z]) / distance_to_goal
        else:
            next_position = self.calculate_new_position(
                robot_trajectory, current_position, goal_point, combined_obstacles, combined_obstacles_radius,
                polyhedron_obstacles, polyhedron_obstacle_radius, polyhedron_obstacle_distance, velocity, time_step
            )

        robot_trajectory_new = np.vstack((robot_trajectory, next_position))
        
        if (self.strekeAB(next_position, goal_point) < Rgoal):
            is_goal_reached = True

        return robot_trajectory_new, is_goal_reached
    
    def calculate_new_position(
        self, robot_trajectory, current_position, goal_point, obstacles, obstacles_radius,
        polyhedron_obstacles, polyhedron_obstacle_radius, polyhedron_obstacle_distance, velocity, dt
    ):
        distance_to_obstacles = np.linalg.norm(current_position - obstacles, axis=1)
        
        if len(polyhedron_obstacles) != 0:
            obstacles = np.vstack((obstacles, polyhedron_obstacles))
            obstacles_radius = np.concatenate((obstacles_radius, polyhedron_obstacle_radius))
            distance_to_obstacles = np.concatenate((distance_to_obstacles, polyhedron_obstacle_distance))

        # Identifying obstacles in a local area
        local_obstacles_indices = np.where(distance_to_obstacles < self.local_radius)[0]
        local_obstacle_radius = obstacles_radius[local_obstacles_indices]
        local_obstacles = obstacles[local_obstacles_indices]
        distance_to_local_obstacles = distance_to_obstacles[local_obstacles_indices] - local_obstacle_radius
        
        nk = np.where(distance_to_local_obstacles < self.Rpred)[0]
        delta_obs = np.zeros(len(local_obstacles_indices))

        if nk.size == 0:  # If there are no obstacles inside the sphere of radius Rpred
            # Set the maximum coefficient of approach to the target
            approach_coefficient = 1
        else:
            D0nk = distance_to_local_obstacles[nk]
            min_distance_to_obstacle = np.min(D0nk)

            if min_distance_to_obstacle < self.Rpred_2:
                approach_coefficient = 0.01
                inv_distances = np.where(D0nk == 0, 0, 1 / D0nk)
                ksi = (1 - approach_coefficient) / np.sum(inv_distances)
                delta_obs[nk] = ksi * inv_distances
            else:
                # The nearest obstacle is located in the ring between spheres of radii Rpred and Rpred_2
                distance_to_goal = self.strekeAB(current_position, goal_point)
                if distance_to_goal < self.RpredB:
                    # Condition for switching to the mode of increased attraction to the target point
                    approach_coefficient = 0.7 if distance_to_goal <= min_distance_to_obstacle else 0.6
                    inv_distances = np.where(D0nk == 0, 0, 1 / D0nk)
                    ksi = (1 - approach_coefficient) / np.sum(inv_distances)
                    delta_obs[nk] = ksi * inv_distances
                else:
                    approach_coefficient = 0.5
                    inv_distances = np.where(distance_to_local_obstacles == 0, 0, 1 / distance_to_local_obstacles)
                    ksi = (1 - approach_coefficient) / np.sum(inv_distances)
                    delta_obs = ksi * inv_distances

        if approach_coefficient == 1:
            # If there are no obstacles inside the sphere of radius Rpred
            force_x = goal_point[0] - current_position[0]
            force_y = goal_point[1] - current_position[1]
            force_z = goal_point[2] - current_position[2]
            
            total_force = np.sqrt(force_x ** 2 + force_y ** 2 + force_z ** 2)
            next_position = current_position + dt * velocity * np.array([force_x, force_y, force_z]) / total_force
        else:
            dividing = np.divide(
                local_obstacle_radius, distance_to_obstacles[local_obstacles_indices],
                out=np.zeros_like(local_obstacle_radius),
                where=(distance_to_obstacles[local_obstacles_indices] != 0)
            )
            local_obstacles_actual = local_obstacles.copy()

            for j in range(local_obstacles.shape[1]):
                local_obstacles_actual[j, :] = local_obstacles[j, :] + dividing[j] * (current_position - local_obstacles[j, :])

            x_tilde = np.sum(delta_obs * local_obstacles_actual[:, 0]) / (1 - approach_coefficient)
            y_tilde = np.sum(delta_obs * local_obstacles_actual[:, 1]) / (1 - approach_coefficient)
            z_tilde = np.sum(delta_obs * local_obstacles_actual[:, 2]) / (1 - approach_coefficient)

            force_x = (x_tilde - current_position[0]) * (1 - approach_coefficient) - approach_coefficient * (
                goal_point[0] - current_position[0]) * self.RB / self.strekeAB(current_position, goal_point)
            force_y = (y_tilde - current_position[1]) * (1 - approach_coefficient) - approach_coefficient * (
                goal_point[1] - current_position[1]) * self.RB / self.strekeAB(current_position, goal_point)
            force_z = (z_tilde - current_position[2]) * (1 - approach_coefficient) - approach_coefficient * (
                goal_point[2] - current_position[2]) * self.RB / self.strekeAB(current_position, goal_point)

            total_force = np.sqrt(force_x ** 2 + force_y ** 2 + force_z ** 2)
            next_position = current_position - dt * velocity * np.array([force_x, force_y, force_z]) / total_force
            
        return next_position
    
    def strekeAB(self, A, B):
        if np.isscalar(A) and np.isscalar(B):
            if B == 0:
                return np.linalg.norm(A)
            else:
                return np.linalg.norm(A - B)
        else:
            if np.all(B == 0):
                return np.sqrt(np.sum(np.square(A)))
            else:
                return np.sqrt(np.sum(np.square(A - B)))
