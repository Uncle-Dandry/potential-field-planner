import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve_banded
from scipy.interpolate import splprep, splev

def distance(A, B):
  return np.linalg.norm(A - B)

def check_segment_intersection(A, B, P1, P2):
  epsilon = 1e-15
  alpha = B - A
  gamma = P2 - P1
  determinant = np.linalg.norm(np.cross(alpha, gamma))

  if abs(determinant) < epsilon:
    return False, None, None, None

  t = np.linalg.norm(np.cross(P1 - A, gamma)) / determinant

  if t < 0 or t > 1:
    return False, None, None, None

  intersection = A + t * alpha

  if distance(intersection, P1) > distance(P1, P2) or distance(intersection, P2) > distance(P1, P2):
    return False, None, None, None

  if distance(P1, intersection) < epsilon:
    return False, None, None, None

  return True, intersection, gamma, np.cross(alpha, gamma)

def check_path_intersection(vertices_build, faces_build, path_x, path_y, path_z):
  path_points = np.vstack((path_x, path_y, path_z)).T

  for obs_idx in range(len(vertices_build)):
    vertices = vertices_build[obs_idx]
    faces = faces_build[obs_idx]
    
    for i in range(len(path_points) - 1):
      A = path_points[i]
      B = path_points[i + 1]
      
      for face in faces:
        if np.any(np.isnan(face)):
          continue

        face_vertices = vertices[face.astype(int)]

        for j in range(len(face_vertices)):
          P1 = face_vertices[j]
          P2 = face_vertices[(j + 1) % len(face_vertices)]
          intersect, _, _, _ = check_segment_intersection(A, B, P1, P2)

          if intersect:
            return True
  return False

def linear_interpolate(s, x, s_new):
  """Vectorized linear interpolation using NumPy."""
  return np.interp(s_new, s, x)

def smooth_trajectory(path):
  """Smooth the trajectory using cubic B-splines."""
  num_points = path.shape[1]
  tck, _ = splprep(path, s=0.5, k=3)
  s_new = np.linspace(0, 1, num_points)
  x_smooth, y_smooth, z_smooth = splev(s_new, tck)
  return np.array(x_smooth), np.array(y_smooth), np.array(z_smooth)

def smooth_path(initial_path, vertices_build, faces_build, obstacle_heights):
  x_initial = initial_path[:, 0]
  y_initial = initial_path[:, 1]
  z_initial = initial_path[:, 2]

  num_inner = len(x_initial) - 2
  acceptable_deviation = 1.5
  delta_steps = 300
  delta_increment = 1 / delta_steps
  num_points = num_inner
  
  x_optimized = []
  y_optimized = []
  z_optimized = []

  if num_points >= 2:
    path_found = False
    deviation = 0

    for u in range(2, delta_steps):
      delta1 = (u - 1) * delta_increment
      delta2 = 1 - delta1
      diag_value = delta1 + 2 * delta2

      if num_points > 2:
        main_diag = np.full(num_points, diag_value)
        off_diag = np.full(num_points - 1, -delta2)
        A_banded = np.zeros((3, num_points))
        A_banded[0, 1:] = off_diag
        A_banded[1] = main_diag
        A_banded[2, :-1] = off_diag

        B_x = delta1 * x_initial[1:-1]
        B_y = delta1 * y_initial[1:-1]
        B_z = delta1 * z_initial[1:-1]
        B_x[0] += delta2 * x_initial[0]
        B_y[0] += delta2 * y_initial[0]
        B_z[0] += delta2 * z_initial[0]
        B_x[-1] += delta2 * x_initial[-1]
        B_y[-1] += delta2 * y_initial[-1]
        B_z[-1] += delta2 * z_initial[-1]

        inner_x = solve_banded((1, 1), A_banded, B_x)
        inner_y = solve_banded((1, 1), A_banded, B_y)
        inner_z = solve_banded((1, 1), A_banded, B_z)

        x_optimized = np.concatenate(([x_initial[0]], inner_x, [x_initial[-1]]))
        y_optimized = np.concatenate(([y_initial[0]], inner_y, [y_initial[-1]]))
        z_optimized = np.concatenate(([z_initial[0]], inner_z, [z_initial[-1]]))
      else:
        A = np.array([[diag_value, -delta2], [-delta2, diag_value]])
        B_x = np.array([delta1 * x_initial[1] + delta2 * x_initial[0], delta1 * x_initial[-2] + delta2 * x_initial[-1]])
        B_y = np.array([delta1 * y_initial[1] + delta2 * y_initial[0], delta1 * y_initial[-2] + delta2 * y_initial[-1]])
        B_z = np.array([delta1 * z_initial[1] + delta2 * z_initial[0], delta1 * z_initial[-2] + delta2 * z_initial[-1]])

        x_optimized = np.concatenate(([x_initial[0]], np.linalg.solve(A, B_x), [x_initial[-1]]))
        y_optimized = np.concatenate(([y_initial[0]], np.linalg.solve(A, B_y), [y_initial[-1]]))
        z_optimized = np.concatenate(([z_initial[0]], np.linalg.solve(A, B_z), [z_initial[-1]]))

      if len(x_optimized) != len(x_initial):
        continue

      deviation = np.sqrt(
        np.sum((x_optimized[1:-1] - x_initial[1:-1])**2 + (y_optimized[1:-1] - y_initial[1:-1])**2 + (z_optimized[1:-1] - z_initial[1:-1])**2)
      ) / np.sqrt(num_inner)

      if deviation <= acceptable_deviation and not check_path_intersection(vertices_build, faces_build, x_optimized, y_optimized, z_optimized):
        path_found = True
        x_optimized, y_optimized, z_optimized = smooth_trajectory(np.vstack((x_optimized, y_optimized, z_optimized)))

        if check_path_intersection(vertices_build, faces_build, x_optimized, y_optimized, z_optimized):
          continue

        break

    initial_length = np.sum([
      distance(np.array([x_initial[i], y_initial[i], z_initial[i]]),
               np.array([x_initial[i + 1], y_initial[i + 1], z_initial[i + 1]]))
      for i in range(len(x_initial) - 1)
    ])
    optimized_length = np.sum([
      distance(np.array([x_optimized[i], y_optimized[i], z_optimized[i]]),
               np.array([x_optimized[i + 1], y_optimized[i + 1], z_optimized[i + 1]]))
      for i in range(len(x_optimized) - 1)
    ])
  
  return [x_optimized, y_optimized, z_optimized]
