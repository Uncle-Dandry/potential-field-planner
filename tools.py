import numpy as np

class PolygonGenerator:
    def __init__(self, max_x, max_y, max_z):
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def generate_polygon_points(
        self, num_polygons, predefined_obstacles_xx, predefined_obstacles_yy, predefined_obstacle_heights,
        start_points, end_points, min_num_vertices=4, max_num_vertices=8, min_distance=7
    ):
        xx_list = []
        yy_list = []
        base_heights = np.zeros(num_polygons + 1)
        obstacle_heights = np.zeros(num_polygons + 1)
        initial_x_offset = 0.04 * self.max_x
        initial_y_offset = 0.08 * self.max_y

        if predefined_obstacles_xx is not None and predefined_obstacles_yy is not None:
            xx_list.extend(predefined_obstacles_xx)
            yy_list.extend(predefined_obstacles_yy)

            if predefined_obstacle_heights is not None:
                obstacle_heights[:len(predefined_obstacle_heights)] = predefined_obstacle_heights

        def is_far_enough(new_building, existing_points, min_distance):
            for point in existing_points:
                point_2d = point[:2]

                for vertex in new_building:
                    if np.linalg.norm(vertex - point_2d) < min_distance:
                        return False
                
                for i in range(len(new_building)):
                    start_vertex = new_building[i]
                    end_vertex = new_building[(i + 1) % len(new_building)]

                    if point_line_distance(point_2d, start_vertex, end_vertex) < min_distance:
                        return False

            return True

        def point_line_distance(point, start, end):
            line_vec = end - start
            point_vec = point - start
            line_len = np.dot(line_vec, line_vec)

            if line_len == 0:
                return np.linalg.norm(point - start)

            t = np.dot(point_vec, line_vec) / line_len
            t = np.clip(t, 0, 1)
            projection = start + t * line_vec

            return np.linalg.norm(point - projection)

        generated_buildings = []

        for i in range(len(predefined_obstacles_xx), num_polygons):
            num_vertices = np.random.randint(min_num_vertices, max_num_vertices + 1)
            valid_position = False
            
            while not valid_position:
                mm_2 = []
                a = initial_x_offset + initial_x_offset * np.random.rand()
                b = initial_y_offset + 1.2 * initial_y_offset * np.random.rand()

                alfa = np.sign(-1 + 2 * np.random.rand()) * np.pi * np.random.rand() / 2
                dsm = np.array([0.5 * (a + b) + np.random.rand() * (0.95 * self.max_x - 0.5 * (a + b)),
                                0.5 * (a + b) + np.random.rand() * (0.95 * self.max_y - 0.5 * (a + b))])
                angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
                points = np.column_stack([np.cos(angles), np.sin(angles)]) * max(a, b)

                for j in range(points.shape[0]):
                    R = np.array([[np.cos(alfa), -np.sin(alfa)],
                        [np.sin(alfa), np.cos(alfa)]])
                    mm_2.append(np.dot(R, points[j]) + dsm)

                new_building = np.array(mm_2)
                new_building_height = np.clip(self.max_z * np.random.rand(), a_min=10, a_max=self.max_z)
                
                all_points = start_points + end_points + [vertex for building in generated_buildings for vertex in building]

                if is_far_enough(new_building, all_points, min_distance):
                    valid_position = True
                    generated_buildings.append(new_building)
            
            xx_list.append(new_building[:, 0])
            yy_list.append(new_building[:, 1])
            base_heights[i] = 0
            obstacle_heights[i] = new_building_height

        # Completion of the underlying surface
        xx_list.append([0, self.max_x, self.max_x, 0])
        yy_list.append([0, 0, self.max_y, self.max_y])
        base_heights[num_polygons] = -10
        obstacle_heights[num_polygons] = 5

        return xx_list, yy_list, base_heights, obstacle_heights

def convert_params_to_vertices_and_faces_single(xx, yy, base_height, obstacle_height):
    num_vertices = len(xx)
    full_height = base_height + obstacle_height
    
    vertices1 = np.column_stack((xx, yy, np.full(num_vertices, base_height)))
    vertices2 = np.column_stack((xx, yy, np.full(num_vertices, full_height)))
    vertices = np.vstack((vertices1, vertices2))
    
    faces = []
    
    i = np.arange(num_vertices)
    i2 = np.arange(num_vertices, 2 * num_vertices)

    for l in range(num_vertices - 1):
        faces.append([i[l], i[l + 1], i2[l + 1], i2[l]])

    faces.append([i[num_vertices - 1], i[0], i2[0], i2[num_vertices - 1]])

    if len(faces[0]) < num_vertices:
        faces = np.column_stack((faces, np.full((len(faces), num_vertices - len(faces[0])), np.nan)))

    faces = np.vstack(([np.arange(num_vertices), np.arange(num_vertices, 2 * num_vertices)], faces))

    return faces, vertices

def convert_params_to_vertices_and_faces(xx_list, yy_list, base_heights, obstacle_heights):
    faces_list = []
    vertices_list = []

    for k in range(len(obstacle_heights)):
        faces, vertices = convert_params_to_vertices_and_faces_single(xx_list[k], yy_list[k], base_heights[k], obstacle_heights[k])
        faces_list.append(faces)
        vertices_list.append(vertices)

    return faces_list, vertices_list

def distance_to_point(A, B):
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

def generate_start_and_end_points(field_sizes, num_robots, start_point, end_point, min_distance_between_robots):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    start_points = [start_point.copy()]
    end_points = [end_point.copy()]
    
    for i in range(1, num_robots):
        offset = np.zeros_like(start_point)
        offset[i % 3] = min_distance_between_robots * (i // 3 + 1)
        new_start_point = start_point + offset
        
        new_start_point = np.minimum(np.maximum(new_start_point, np.zeros_like(new_start_point)), field_sizes)
        start_points.append(new_start_point)
        new_end_point = end_point + offset

        new_end_point = np.minimum(np.maximum(new_end_point, np.zeros_like(new_end_point)), field_sizes)
        end_points.append(new_end_point)
        
    return start_points, end_points
