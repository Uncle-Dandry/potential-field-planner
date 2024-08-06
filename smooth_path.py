import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve

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
  x_smooth = np.zeros(len(s_new))

  for i in range(len(s_new)):
    for j in range(len(s) - 1):
      if s[j] <= s_new[i] <= s[j + 1]:
        t = (s_new[i] - s[j]) / (s[j + 1] - s[j])
        x_smooth[i] = (1 - t) * x[j] + t * x[j + 1]
        break

  return x_smooth

def smooth_trajectory(path):
  x = path[0]
  y = path[1]
  z = path[2]
  s = np.zeros(len(x))

  for i in range(1, len(x)):
    s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2 + (z[i] - z[i - 1]) ** 2)

  length = s[-1]
  s_new = np.linspace(0, length, len(x))
  x_smooth = linear_interpolate(s, x, s_new)
  y_smooth = linear_interpolate(s, y, s_new)
  z_smooth = linear_interpolate(s, z, s_new)
  
  return x_smooth, y_smooth, z_smooth

def smooth_path(initial_path, vertices_build, faces_build, obstacle_heights):
  x_initial = initial_path[:, 0]
  y_initial = initial_path[:, 1]
  z_initial = initial_path[:, 2]

  N = len(x_initial) - 2
  acceptable_deviation = 1.5
  delta_steps = 300
  delta_increment = 1 / delta_steps
  num_points = N - 2
  
  x_optimized = []
  y_optimized = []
  z_optimized = []

  if num_points >= 2:
    A = np.zeros((num_points, num_points))
    B_x = np.zeros(num_points)
    B_y = np.zeros(num_points)
    B_z = np.zeros(num_points)

    path_found = False
    deviation = 0

    for u in range(2, delta_steps):
      delta1 = (u - 1) * delta_increment
      delta2 = 1 - delta1
      diag_value = delta1 + 2 * delta2

      if num_points > 2:
        for i in range(num_points):
          A[i, i] = diag_value
          if i == 0:
            A[i, i + 1] = -delta2
            B_x[i] = delta1 * x_initial[1] + delta2 * x_initial[0]
            B_y[i] = delta1 * y_initial[1] + delta2 * y_initial[0]
            B_z[i] = delta1 * z_initial[1] + delta2 * z_initial[0]
          elif i == num_points - 1:
            A[i, i - 1] = -delta2
            B_x[i] = delta1 * x_initial[N - 2] + delta2 * x_initial[N - 1]
            B_y[i] = delta1 * y_initial[N - 2] + delta2 * y_initial[N - 1]
            B_z[i] = delta1 * z_initial[N - 2] + delta2 * z_initial[N - 1]
          else:
            A[i, i - 1] = -delta2
            A[i, i + 1] = -delta2
            B_x[i] = delta1 * x_initial[i + 1]
            B_y[i] = delta1 * y_initial[i + 1]
            B_z[i] = delta1 * z_initial[i + 1]
      else:
        A = np.array([[diag_value, -delta2], [-delta2, diag_value]])
        B_x = np.array([delta1 * x_initial[1] + delta2 * x_initial[0], delta1 * x_initial[2] + delta2 * x_initial[3]])
        B_y = np.array([delta1 * y_initial[1] + delta2 * y_initial[0], delta1 * y_initial[2] + delta2 * y_initial[3]])
        B_z = np.array([delta1 * z_initial[1] + delta2 * z_initial[0], delta1 * z_initial[2] + delta2 * z_initial[3]])

      x_optimized = np.concatenate(([x_initial[0]], solve(A, B_x), [x_initial[N - 1]]))
      y_optimized = np.concatenate(([y_initial[0]], solve(A, B_y), [y_initial[N - 1]]))
      z_optimized = np.concatenate(([z_initial[0]], solve(A, B_z), [z_initial[N - 1]]))

      if len(x_optimized) != len(x_initial) - 2:
        continue

      deviation = np.sqrt(
        np.sum((x_optimized - x_initial[1:-1])**2 + (y_optimized - y_initial[1:-1])**2 + (z_optimized - z_initial[1:-1])**2)
      ) / np.sqrt(N - 2)

      if deviation <= acceptable_deviation and not check_path_intersection(vertices_build, faces_build, x_optimized, y_optimized, z_optimized):
        path_found = True
        x_optimized, y_optimized, z_optimized = smooth_trajectory(np.vstack((x_optimized, y_optimized, z_optimized)))

        if check_path_intersection(vertices_build, faces_build, x_optimized, y_optimized, z_optimized):
          continue

        break

    initial_length = np.sum([distance(np.array([x_initial[i], y_initial[i], z_initial[i]]),
      np.array([x_initial[i + 1], y_initial[i + 1], z_initial[i + 1]])) for i in range(N - 1)])
    optimized_length = np.sum([distance(np.array([x_optimized[i], y_optimized[i], z_optimized[i]]),
      np.array([x_optimized[i + 1], y_optimized[i + 1], z_optimized[i + 1]])) for i in range(N - 1)])
  
  return [x_optimized, y_optimized, z_optimized]
