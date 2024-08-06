from experiments import Scene

# The following parameters configure the scene. You should change them by creating a scene in the experiments.py file.
max_x = 100
max_y = 100
max_z = 40

start_point = [2, 2, 4]
end_point = [90, 90, 10]

num_robots = 4
min_distance_between_robots = 2.2

num_obstacles = 5
predefined_obstacles_xx = []
predefined_obstacles_yy = []
predefined_obstacles_heights = []

predefined_scene = Scene.DEFAULT

# additional parameters for following, maintaining distance and smoothing
follow = False
maintain_distance = False
enable_smooth = True

# parameters of time step, speed and size of the robot
dt_proc = 1
cruising_speed = 1.5
inner_robot_radius = 0.5

# parameter of the multiplier of the protective radius from the obstacle
obstacle_radius_multiplier = 1.1

# radius parameter of the sphere of the removal limit for maintaining the distance
Rmaxgr = 4

# optimizing parameter of the calculated obstacle avoidance area for UAV
local_radius = 40

# the radii of two nested neighborhoods to determine the vector of virtual forces of attraction
Rpred = 5
Rpred_2 = 0.3 * Rpred
