# Potential Field Planner for Multi-Robot Coordination

## Overview

This repository contains a Python implementation of a Potential Field Planner for multi-robot trajectory planning. The planner is designed to coordinate the movement of a group of robots in a three-dimensional environment, avoiding obstacles and reaching target points. The implementation demonstrates fundamental concepts of potential fields, where robots are attracted to target points and repelled from obstacles, including other robots.

**Note:** This project is an amateur implementation for educational purposes and is not intended for production use.

## Features

- **Trajectory Planning:** Plans safe paths for multiple robots from start to goal points, avoiding obstacles.
- **Obstacle Avoidance:** Models obstacles as spherical repulsion fields to prevent collisions.
- **Local Minimum Handling:** Detects and escapes local minima situations where robots are trapped.
- **Group Behavior:** Supports leader-following (`follow`) and distance maintenance (`maintain_distance`) within a robot group.
- **Random Scene Generation:** Includes a feature (`random_obs`) to generate random obstacles in the environment (works imperfectly).
- **3D Visualization:** Provides visualization of the planned trajectories in a 3D environment.

## Installation

To use the Potential Field Planner, you need Python installed. The following Python packages are required:

- `numpy`
- `matplotlib`
- `mpl_toolkits.mplot3d`

Install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/potential-field-planner.git
   cd potential-field-planner
   ```

2. **Configure the Simulation:**
   Edit the `config.py` file to set the parameters of the simulation, such as the number of robots, obstacle positions, start and goal points, and other settings. 

3. **Run Predefined Experiments:**
   The file `experiments.py` includes several predefined scenarios to demonstrate the capabilities of the planner. Run it using:
   ```bash
   python experiments.py
   ```

4. **Run a Custom Simulation:**
   You can create a custom scenario or experiment by modifying the settings in `config.py` or directly within your Python scripts.

5. **Random Scene Generation:**
   Use the `random_obs` flag to generate a random scene with obstacles. Note that this feature is not fully reliable and may produce unexpected results.

6. **Group Behavior Control:**
   - Use the `follow` flag to enable a leader-following behavior among the robots.
   - Use the `maintain_distance` flag to enforce distance maintenance within the group.

## Code Structure

- `main.py`: The main script to run custom simulations.
- `config.py`: Configuration file for setting parameters.
- `potential_field_planner.py`: Core implementation of the potential field algorithm.
- `visualization.py`: Functions for visualizing robots' trajectories and the environment.
- `utils.py`: Utility functions for calculations and helpers.
- `experiments.py`: Script containing predefined scenarios for demonstration.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Make sure to test your changes and provide a description.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is a result of a research initiative to explore multi-robot coordination using potential fields. It is intended for educational and demonstration purposes.

---

For more detailed information, please refer to the documentation provided in the repository.
