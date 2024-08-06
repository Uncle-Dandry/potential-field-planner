import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def create_polygons(vertices, faces):
  polygons = []

  for face in faces:
    polygon = []

    for index in face:
      if not np.isnan(index):
        polygon.append(vertices[int(index)])

    polygons.append(polygon)

  return polygons

def create_sphere(center, radius=1, resolution=20):
  phi, theta = np.mgrid[0.0:2.0*np.pi:complex(resolution), 0.0:np.pi:complex(resolution)]
  x = radius * np.sin(theta) * np.cos(phi) + center[0]
  y = radius * np.sin(theta) * np.sin(phi) + center[1]
  z = radius * np.cos(theta) + center[2]

  return x, y, z

def plot_3d_graph(
  num_robots, start_points, end_points, obstacle_heights, vertices_build, faces_build,
  robot_trajectories, enable_smooth, field_sizes, strong_traj=False
):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.grid(False)

  for point_index in range(len(start_points)):
    x, y, z = create_sphere(start_points[point_index], radius=2)
    ax.plot_surface(x, y, z, color='b', alpha=0.7)

    x, y, z = create_sphere(end_points[point_index], radius=2)
    ax.plot_surface(x, y, z, color='r', alpha=0.7)

  max_x, max_y, max_z = field_sizes
  
  if obstacle_heights.any():
    for k in range(len(obstacle_heights) - 1):
      vertices = vertices_build[k]
      faces = faces_build[k]
      polygons = create_polygons(vertices, faces)
      
      for polygon in polygons:
        poly3d = [polygon]
        collection = Poly3DCollection(poly3d, facecolors='g', edgecolors='k', alpha=0.8, linewidths=0.2)
        collection.set_zsort('max')
        ax.add_collection3d(collection)

  ax.set_xlim(0, max_x)
  ax.set_ylim(0, max_y)
  ax.set_zlim(0, max_z)

  for robot_index in range(num_robots):
    if enable_smooth:
      x_vals = robot_trajectories[robot_index][0]
      y_vals = robot_trajectories[robot_index][1]
      z_vals = robot_trajectories[robot_index][2]
    else:
      x_vals = [elem[0] for elem in robot_trajectories[robot_index]]
      y_vals = [elem[1] for elem in robot_trajectories[robot_index]]
      z_vals = [elem[2] for elem in robot_trajectories[robot_index]]
    
    segments = [[x_vals[j], y_vals[j], z_vals[j]] for j in range(len(x_vals))]
    color = 'b' if (strong_traj and robot_index == 0) else 'r'
    ax.add_collection3d(Line3DCollection([segments], colors=color, linewidths=1.6, alpha=1.0))

  ax.view_init(elev=20, azim=30)
  
  plt.show()
  