import numpy as np

from config import (
  max_x, max_y, max_z,
  start_point, end_point, num_robots, min_distance_between_robots,
  num_obstacles, predefined_obstacles_xx, predefined_obstacles_yy, predefined_obstacles_heights,
  follow, maintain_distance, enable_smooth
)
  
from experiments import Scene, getSceneData
from graphics import plot_3d_graph
from potential_field_planner import PotentialFieldPlanner
from smooth_path import smooth_path
from tools import PolygonGenerator, generate_start_and_end_points, convert_params_to_vertices_and_faces

max_x, max_y, max_z, num_obstacles, predefined_obstacles_xx, predefined_obstacles_yy, predefined_obstacles_heights, start_point, end_point, num_robots = getSceneData(Scene.DEFAULT)

start_points, end_points = generate_start_and_end_points((max_x, max_y, max_z), num_robots, start_point, end_point, min_distance_between_robots)

polygon_generator = PolygonGenerator(max_x, max_y, max_z)
obstacles_xx, obstacles_yy, base_heights, obstacles_heights = polygon_generator.generate_polygon_points(
  num_obstacles, predefined_obstacles_xx, predefined_obstacles_yy, predefined_obstacles_heights, start_points, end_points)

# print(obstacles_xx, obstacles_yy, obstacles_heights) # if need get obstacles data

def main(max_x, max_y, max_z, start_points, end_points, num_robots,
     obstacles_xx, obstacles_yy, base_heights, obstacles_heights):
  faces_build, vertices_build = convert_params_to_vertices_and_faces(obstacles_xx, obstacles_yy, base_heights, obstacles_heights)

  potential_field_planner = PotentialFieldPlanner(num_robots, max_x, max_y, max_z, follow_leader=follow, maintain_distance=maintain_distance)
  trajectories, target_reached_flags = potential_field_planner.plan_group_trajectories(np.array(start_points).T, np.array(end_points).T,
    obstacles_heights, vertices_build, faces_build, (max_x, max_y, max_z))

  if enable_smooth:
    for robot_num in range(num_robots):
      trajectories[robot_num] = smooth_path(trajectories[robot_num], vertices_build, faces_build, obstacles_heights)

  plot_3d_graph(num_robots, start_points, end_points, obstacles_heights, vertices_build, faces_build,
    trajectories, enable_smooth, (max_x, max_y, max_z), strong_traj=(follow or maintain_distance))

if __name__ == "__main__":
  main(max_x, max_y, max_z, start_points, end_points, num_robots,
     obstacles_xx, obstacles_yy, base_heights, obstacles_heights)
