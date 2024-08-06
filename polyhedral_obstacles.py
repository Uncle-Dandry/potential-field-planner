import numpy as np

from tools import distance_to_point

def create_main_obstacle(max_x, max_y, obstacle_radius_multiplier):
    real_radius = 0.5 * (max_x + max_y) * 15
    radius = real_radius * obstacle_radius_multiplier
    center = np.array([[0.5 * max_x], [0.5 * max_y], [-radius]])

    return radius, center.flatten()

def compute_nearest_obstacle_from_point(faces, vertices, point):
    delta_accuracy = 1e-8
    min_distance_to_edge, min_distance_to_face = np.inf, np.inf
    nearest_obstacle_point, XEmean, XFmean = None, None, None
    
    distances_to_vertices = np.sqrt(np.sum((vertices - point) ** 2, axis=1))
    min_distance_to_vertex_index = np.argmin(distances_to_vertices)
    min_distance_to_vertex = np.min(distances_to_vertices)

    for i in range(faces.shape[0]):
        for j in range(faces.shape[1] - 1):
            num1, num2 = 0, 0

            if not np.isnan(faces[i, j]) and not np.isnan(faces[i, j + 1]):
                num1, num2 = int(faces[i, j]), int(faces[i, j + 1])
            elif not np.isnan(faces[i, j]) and np.isnan(faces[i, j + 1]):
                num1, num2 = int(faces[i, 0]), int(faces[i, j])
                
            if num1 != 0:
                S = 0.5 * distance_to_point(np.cross(point - vertices[num1], point - vertices[num2]), 0)
                h = 2 * S / distance_to_point(vertices[num1], vertices[num2])
                d1 = distances_to_vertices[num1]
                d2 = distances_to_vertices[num2]
                
                alfa = np.arccos(h / d1)
                beta = np.arccos(h / d2)
                fi = np.arccos(np.dot(vertices[num1] - point, vertices[num2] - point) / (d1 * d2))
                
                if np.abs(fi - (alfa + beta)) < delta_accuracy and h <= min_distance_to_edge:
                    min_distance_to_edge = h
                    XEmean = vertices[num1] + d1 * np.sin(alfa) * (vertices[num2] - vertices[num1]) / distance_to_point(vertices[num2], vertices[num1])
                
            if np.isnan(faces[i, j + 1]):
                break
                
        A, B, C = vertices[int(faces[i, 0])], vertices[int(faces[i, 1])], vertices[int(faces[i, 2])]
        a, b, c = np.cross(B - A, C - A)
        
        Pproec_plane = -np.dot([a, b, c], point - A) / (a**2 + b**2 + c**2)
        Pproec = point + np.array([a, b, c]) * Pproec_plane
        teta = 0

        for k in range(faces.shape[1] - 1):
            num1, num2 = 0, 0

            if not np.isnan(faces[i, k]) and not np.isnan(faces[i, k + 1]):
                num1, num2 = int(faces[i, k]), int(faces[i, k + 1])
            elif not np.isnan(faces[i, k]) and np.isnan(faces[i, k + 1]):
                num1, num2 = int(faces[i, 0]), int(faces[i, k])

            teta += np.arccos(np.dot(vertices[num1] - Pproec, vertices[num2] - Pproec) / (distance_to_point(vertices[num1], Pproec) * distance_to_point(vertices[num2], Pproec)))

            if np.isnan(faces[i, k + 1]):
                break
    
        if not np.isnan(faces[i, faces.shape[1] - 1]):
            num1, num2 = int(faces[i, faces.shape[1] - 1]), int(faces[i, 0])
            teta += np.arccos(np.dot(vertices[num1] - Pproec, vertices[num2] - Pproec) / (distance_to_point(vertices[num1], Pproec) * distance_to_point(vertices[num2], Pproec)))

        if np.abs(teta - 2 * np.pi) < delta_accuracy:
            d = distance_to_point(point, Pproec)

            if d <= min_distance_to_face:
                min_distance_to_face = d
                XFmean = Pproec

    if min_distance_to_edge <= min_distance_to_face and min_distance_to_edge <= min_distance_to_vertex:
        min_distance = min_distance_to_edge
        nearest_obstacle_point = XEmean
    elif min_distance_to_face <= min_distance_to_edge and min_distance_to_face <= min_distance_to_vertex:
        min_distance = min_distance_to_face
        nearest_obstacle_point = XFmean
    else:
        min_distance = min_distance_to_vertex
        nearest_obstacle_point = vertices[min_distance_to_vertex_index]

    return nearest_obstacle_point, min_distance

def translate_polyhedrons_to_obs(start_point, faces_build, vertices_build, cruising_speed, dt_proc):
    obstacles, obstacles_radius, distances_to_obstacles = [], [], []

    for i in range(len(faces_build)):
        line_min = cruising_speed * dt_proc
        obstacle_center_point, obstacle_distance = compute_nearest_obstacle_from_point(faces_build[i], vertices_build[i], start_point)
        
        obstacles.append(obstacle_center_point)
        obstacles_radius.append(min(0.9 * obstacle_distance, line_min))
        distances_to_obstacles.append(obstacle_distance)

    return obstacles, obstacles_radius, distances_to_obstacles
