import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def import_obstacle_config(gen_xml_path):
    """
    Dynamically imports the OBSTACLE_CONFIG dictionary from gen_xml.py.
    This is far more robust than parsing the file with regex.
    """
    try:
        # Temporarily add the directory of gen_xml.py to the system path
        # so we can import it like a module.
        spec_dir = os.path.dirname(gen_xml_path)
        sys.path.insert(0, spec_dir)

        # Import the specific variable from the module
        from gen_xml import OBSTACLE_CONFIG
        
        # Clean up the system path
        sys.path.pop(0)

        print(f"Successfully imported config from gen_xml.py:")
        print(f"  - {OBSTACLE_CONFIG}")
        
        return OBSTACLE_CONFIG
        
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not import OBSTACLE_CONFIG from {gen_xml_path}.")
        print(f"Please ensure gen_xml.py exists and contains an 'OBSTACLE_CONFIG' dictionary.")
        print(f"Details: {e}")
        return None


def generate_maneuver_plan(config):
    """
    Generates a more realistic footstep plan: straight walking with
    discrete "lane change" maneuvers to avoid each obstacle.
    """
    # --- Path & Maneuver Parameters ---
    amplitude = 0.6      # How far to swing sideways (Y-axis) for evasion.
    maneuver_distance = 3.0 # The distance over which the lane change happens (X-axis).
    step_length = 0.3    # Forward step distance.
    stance_width = 0.2   # Lateral distance between feet.
    yaw_threshold = 0.1  # (radians) If yaw is less than this, treat as straight (0.0).

    footsteps = [[0.0, -stance_width / 2.0, 0.0]] # Initial right foot position
    current_x = 0.0
    current_y = 0.0
    foot_side = 1 # Start with the left foot (+1).
    evasion_side = 1 # Start by evading to the right (+1).

    # --- Loop through each obstacle to plan the path ---
    for i in range(config['num']):
        obstacle_x = config['start_x'] + i * config['spacing_x']
        
        # Define the start and end of the evasion maneuver for this obstacle
        start_maneuver_x = obstacle_x - maneuver_distance / 2.0
        end_maneuver_x = obstacle_x + maneuver_distance / 2.0

        # --- Phase 1: Straight approach before the maneuver ---
        while current_x + step_length < start_maneuver_x:
            current_x += step_length
            foot_y = foot_side * stance_width / 2.0
            footsteps.append([current_x, foot_y, 0.0]) # Yaw is 0 for straight
            foot_side *= -1

        # --- Phase 2: The evasion maneuver (a single sine-wave bump) ---
        # This loop generates the S-curve footsteps
        while current_x < end_maneuver_x:
            # Calculate the target y-position on the sine wave
            progress = (current_x - start_maneuver_x) / maneuver_distance
            target_y = evasion_side * amplitude * np.sin(progress * np.pi)
            
            # Calculate the next footstep position
            next_x = current_x + step_length
            next_progress = (next_x - start_maneuver_x) / maneuver_distance
            next_target_y = evasion_side * amplitude * np.sin(next_progress * np.pi)

            # The foot lands halfway between the current and next target y, plus stance
            foot_y = (target_y + next_target_y) / 2.0 + (foot_side * stance_width / 2.0)
            
            # Calculate yaw based on the direction of the curve
            delta_x = next_x - current_x
            delta_y = next_target_y - target_y
            yaw = np.arctan2(delta_y, delta_x)

            # *** THIS IS THE KEY CHANGE ***
            # If the turn angle is very small, just walk straight.
            if abs(yaw) < yaw_threshold:
                yaw = 0.0
            
            current_x = next_x
            footsteps.append([current_x, foot_y, yaw])
            foot_side *= -1

        # Flip the evasion side for the next obstacle
        evasion_side *= -1

    # --- Phase 3: Final straight walk after the last obstacle ---
    for _ in range(10): # Add 10 more straight steps
        current_x += step_length
        # Ensure the robot returns to the center line
        foot_y = (foot_side * stance_width / 2.0)
        footsteps.append([current_x, foot_y, 0.0])
        foot_side *= -1

    return footsteps




def save_plan_to_file(footsteps, output_path):
    """Saves the footstep plan to the specified file."""
    try:
        with open(output_path, 'w') as f:
            f.write("---\n")
            for step in footsteps:
                f.write(f"{step[0]},{step[1]},{step[2]}\n")
        print(f"\nSuccessfully generated and saved new footstep plan to:\n{output_path}")
    except IOError as e:
        print(f"\nError: Could not write to file {output_path}. Reason: {e}")

def visualize_plan(footsteps, config):
    """Visualizes the generated footstep plan."""
    footsteps_np = np.array(footsteps)
    plt.figure(figsize=(15, 4))
    plt.plot(footsteps_np[:, 0], footsteps_np[:, 1], '.-', label='Footstep Path')
    for i in range(config['num']):
        obstacle_x = config['start_x'] + i * config['spacing_x']
        obstacle_radius = config.get('radius', 0.2)
        circle = plt.Circle((obstacle_x, 0), obstacle_radius, color='r', label='Obstacle' if i == 0 else "")
        plt.gca().add_patch(circle)
    plt.title('Generated Footstep Plan vs. Obstacles')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Add project root to sys.path so that gen_xml can find its dependencies (e.g., 'models')
    sys.path.insert(0, project_root)

    gen_xml_path = os.path.join(project_root, 'envs', 'jvrc',  'gen_xml.py')
    output_path = os.path.join(script_dir, 'footstep_plans.txt')

    # 1. Import the obstacle configuration
    config = import_obstacle_config(gen_xml_path)

    if config:
        # 2. Generate the footstep plan
        plan = generate_maneuver_plan(config)
        
        # 3. Save the plan to the file
        save_plan_to_file(plan, output_path)

        # 4. Visualize the plan
        print("\nDisplaying visual plan for verification...")
        visualize_plan(plan, config)