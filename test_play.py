import time
import mujoco
import mujoco.viewer
import h5py
import numpy as np
import os
import argparse

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="MuJoCo G1 Dataset Player")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="scene.xml", 
        help="Path to the MuJoCo XML model file (default: scene.xml)"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="recorded_demo1.hdf5", 
        help="Path to the HDF5 dataset file"
    )
    
    parser.add_argument(
        "--demo_name", 
        type=str, 
        default="demo_0", 
        help="Name of the demo key in the HDF5 file"
    )
    
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.0, 
        help="Playback speed multiplier (1.0 = real-time)"
    )

    args = parser.parse_args()
    
    # Use arguments
    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset
    DEMO_NAME = args.demo_name
    PLAY_SPEED = args.speed

    # 1. Check files
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please ensure the XML file exists or provide the correct path using --model_path")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file '{DATASET_PATH}' not found.")
        return

    # 2. Load MuJoCo Model
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # 3. Load Dataset
    print(f"Loading dataset: {DATASET_PATH}")
    with h5py.File(DATASET_PATH, 'r') as f:
        # Check if 'data' group exists (common structure) or if keys are at root
        root = f['data'] if 'data' in f else f
        
        if DEMO_NAME not in root:
            print(f"Demo '{DEMO_NAME}' not found.")
            print(f"Available keys: {list(root.keys())}")
            return
            
        demo = root[DEMO_NAME]
        
        # Extract Joint Positions and Root Pose
        # Logic to handle different dataset structures (Isaac Lab vs Custom Recorder)
        
        # Case A: Custom Recorder (record_g1_mocap.py)
        if 'obs' in demo and 'robot_joint_pos' in demo['obs']:
            joint_pos = demo['obs']['robot_joint_pos'][:]
            root_pos = demo['obs']['robot_root_pos'][:]
            root_rot = demo['obs']['robot_root_rot'][:]
            root_pose = np.hstack([root_pos, root_rot])
            
        # Case B: Isaac Lab / Robomimic Structure
        elif 'states' in demo and 'articulation' in demo['states']:
            robot_grp = demo['states']['articulation']['robot'] # May vary based on naming
            # Sometimes 'robot' is a key, sometimes it's the actual name of the robot
            # If 'robot' key doesn't exist, grab the first key in articulation
            if 'robot' not in robot_grp and len(robot_grp.keys()) > 0:
                 first_key = list(robot_grp.keys())[0]
                 robot_grp = robot_grp[first_key]
            elif 'robot' in robot_grp:
                 robot_grp = robot_grp['robot']

            joint_pos = robot_grp['joint_position'][:]
            root_pose = robot_grp['root_pose'][:] 
            
        # Case C: Fallback / Simple qpos dump
        elif 'qpos' in demo:
            full_qpos = demo['qpos'][:]
            root_pose = full_qpos[:, :7]
            joint_pos = full_qpos[:, 7:]
            
        else:
            print("Could not find state data in standard keys (states/obs/qpos).")
            return

    num_frames = joint_pos.shape[0]
    print(f"Found {num_frames} frames of data.")
    print(f"Dataset Joint Dims: {joint_pos.shape[1]}")
    print(f"Model Joint Dims: {model.nq - 7}") # Subtract 7 for free joint (root)

    # 4. Launch Viewer and Play
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Playing...")
        start_time = time.time()
        
        # Determine how many joints to map
        # If dataset has hands (43) but model is body-only (29), we slice.
        joints_to_set = min(joint_pos.shape[1], model.nq - 7)

        for i in range(num_frames):
            if not viewer.is_running():
                break

            # --- Map Data to MuJoCo ---
            
            # 1. Set Root Pose (Pos + Quat)
            data.qpos[:3] = root_pose[i, :3] # x, y, z
            data.qpos[3:7] = root_pose[i, 3:] # w, x, y, z

            # 2. Set Joint Positions
            data.qpos[7 : 7 + joints_to_set] = joint_pos[i, :joints_to_set]

            # 3. Step Simulation (Forward Kinematics only)
            mujoco.mj_forward(model, data)
            
            # 4. Render
            viewer.sync()

            # Timing
            time_step = 0.02 # Assuming 50Hz
            time.sleep(time_step / PLAY_SPEED)

    print("Done.")

if __name__ == "__main__":
    main()