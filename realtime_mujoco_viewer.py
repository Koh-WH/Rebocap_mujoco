#!/usr/bin/env python3
"""
Real-Time ReboCap MuJoCo G1 Viewer
Streams motion data directly from ReboCap SDK and renders in MuJoCo viewer
NO WEBSOCKET NEEDED - Direct SDK connection
Refer to def(main) for arguments.
Run 'python realtime_mujoco_viewer.py --model_path scene.xml --calibrate'
"""

import argparse
import queue
import threading
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys

# Import ReboCap SDK
try:
    import rebocap_ws_sdk
    REBOCAP_SDK_AVAILABLE = True
except ImportError:
    print("❌ ReboCap SDK not found!")
    print("Make sure 'rebocap_ws_sdk' folder is in the same directory")
    sys.exit(1)

# ============================================
# REBOCAP JOINT NAMES (24 joints)
# ============================================
REBOCAP_JOINT_NAMES = rebocap_ws_sdk.REBOCAP_JOINT_NAMES

# ============================================
# COORDINATE SYSTEM CONFIGURATION
# ============================================
# ReboCap Unity coordinates: X=right, Y=up, Z=forward
# G1 MuJoCo coordinates: X=forward, Y=left, Z=up
# We need to remap axes accordingly

COORDINATE_REMAP = {
    'use_remapping': True,  # Set to False to disable remapping
    # Unity -> MuJoCo axis mapping
    # Unity [roll(X), pitch(Y), yaw(Z)] -> MuJoCo [roll, pitch, yaw]
    'axis_map': {
        0: 1,   # Unity X (roll, side) -> MuJoCo Y (pitch, forward/back)
        1: 2,   # Unity Y (pitch, up) -> MuJoCo Z (yaw, twist)
        2: 0    # Unity Z (yaw, forward) -> MuJoCo X (roll, side)
    },
    'axis_signs': {
        0: 1,   # May need to flip
        1: 1,
        2: 1
    }
}

# ============================================
# JOINT MAPPING: ReboCap -> G1 MuJoCo
# ============================================
# Format: rebocap_index: [(joint_name, euler_axis, scale, offset), ...]
# euler_axis: 0=roll(X), 1=pitch(Y), 2=yaw(Z) in Unity coordinates
REBOCAP_TO_G1_MAPPING = {
    # Waist (Sensor: Waist + Chest)
    0: [  # Pelvis
        ('waist_yaw_joint', 2, 1.5, 0.0),
        ('waist_roll_joint', 0, -1.3, 0.0)
    ],
    3: [('waist_pitch_joint', 1, 1.0, 0.0)],  # Spine1
    9: [('waist_pitch_joint', 1, 1.0, 0.0)],  # Spine3 (chest)
    
    # Left Leg (Sensors: Left Thigh + Calf)
    1: [  # L_Hip
        ('left_hip_pitch_joint', 1, 0.5, 0.0),
        ('left_hip_roll_joint', 0, 0.5, 0.0),
        ('left_hip_yaw_joint', 2, 0.5, 0.0)
    ],
    4: [('left_knee_joint', 1, 0.9, 0.0)],  # L_Knee
    
    # Right Leg (Sensors: Right Thigh + Calf)
    2: [  # R_Hip
        ('right_hip_pitch_joint', 1, 0.5, 0.0),
        ('right_hip_roll_joint', 0, -0.5, 0.0),
        ('right_hip_yaw_joint', 2, -0.5, 0.0)
    ],
    5: [('right_knee_joint', 1, 0.9, 0.0)],  # R_Knee
    
    # Left Arm (Sensors: Left Upper Arm + Lower Arm + Hand)
    16: [  # L_Shoulder
        ('left_shoulder_pitch_joint', 2, -1.0, 0.0),
        ('left_shoulder_roll_joint', 0, -1.2, 1.7),
        ('left_shoulder_yaw_joint', 1, -1.2, 0.0)
    ],
    18: [('left_elbow_joint', 0, 0.7, 1.0)],  # L_Elbow
    20: [  # L_Wrist
        ('left_wrist_pitch_joint', 2, -0.9, 0.0),
        ('left_wrist_roll_joint', 1, 0.9, 0.0),
        ('left_wrist_yaw_joint', 0, -0.9, 0.0)
    ],
    
    # Right Arm (Sensors: Right Upper Arm + Lower Arm + Hand)
    17: [  # R_Shoulder
        ('right_shoulder_pitch_joint', 2, 1.0, 0.0),
        ('right_shoulder_roll_joint', 0, -1.2, -1.7),
        ('right_shoulder_yaw_joint', 1, 1.2, 0.0)
    ],
    19: [('right_elbow_joint', 0, -0.7, 1.0)],  # R_Elbow
    21: [  # R_Wrist
        ('right_wrist_pitch_joint', 2, 0.9, 0.0),
        ('right_wrist_roll_joint', 1, -0.9, 0.0),
        ('right_wrist_yaw_joint', 0, -0.9, 0.0)
    ],
}

# ============================================
# REBOCAP RECEIVER (Direct SDK)
# ============================================
class RebocapDirectReceiver:
    """Receives real-time motion data directly from ReboCap SDK"""
    
    def __init__(self, port: int = 7690):
        self.port = port
        self.data_queue = queue.Queue(maxsize=5)
        self.running = False
        self.connected = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Initialize SDK
        self.sdk = rebocap_ws_sdk.RebocapWsSdk(
            coordinate_type=rebocap_ws_sdk.CoordinateType.UnityCoordinate,
            use_global_rotation=False
        )
        
        # Set callback
        self.sdk.set_pose_msg_callback(self._on_pose_callback)
        
    def _on_pose_callback(self, sdk_inst, tran, pose24, static_idx, ts):
        """Callback when new pose data arrives"""
        try:
            # Validate data
            if not pose24 or len(pose24) != 24:
                return
            
            # Package data
            data = {
                "tran": list(tran) if tran else [0, 0, 0],
                "pose24": [list(q) for q in pose24],
                "timestamp": ts
            }
            
            self.frame_count += 1
            self.fps_counter += 1
            
            # Add to queue
            if self.data_queue.full():
                try:
                    self.data_queue.get_nowait()
                except:
                    pass
            self.data_queue.put(data)
            
            # FPS display
            if time.time() - self.last_fps_time >= 3.0:
                fps = self.fps_counter / (time.time() - self.last_fps_time)
                print(f"📊 Receiving: {fps:.1f} FPS | Total frames: {self.frame_count}")
                self.fps_counter = 0
                self.last_fps_time = time.time()
                
        except Exception as e:
            print(f"⚠️  Callback error: {e}")
    
    def start(self):
        """Start receiving data"""
        print(f"🔌 Connecting to ReboCap on port {self.port}...")
        
        result = self.sdk.open(self.port)
        if result == 0:
            self.connected = True
            self.running = True
            print("✅ Connected to ReboCap SDK!")
            print("📡 Waiting for motion data...")
            return True
        else:
            print(f"❌ Failed to connect (error code: {result})")
            print("\nTroubleshooting:")
            print("  1. Make sure ReboCap software is running")
            print("  2. Check that ReboCap is broadcasting on port", self.port)
            print("  3. Verify no other application is using the port")
            return False
    
    def get_latest_data(self):
        """Get latest motion data"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop receiver"""
        self.running = False
        self.connected = False
        try:
            self.sdk.close()
            print("🔌 ReboCap SDK closed")
        except:
            pass

# ============================================
# MOTION CONVERTER
# ============================================
class MotionConverter:
    """Converts ReboCap data to MuJoCo joint positions"""
    
    def __init__(self, model):
        self.model = model
        self.joint_addr_map = self._build_joint_map()
        self.neutral_pose = None
        self.is_calibrated = False
        
        print(f"📋 Joint mapping initialized:")
        print(f"   - G1 joints found: {len(self.joint_addr_map)}")
        print(f"   - ReboCap joints mapped: {len(REBOCAP_TO_G1_MAPPING)}")
    
    def _build_joint_map(self):
        """Build map of joint names to qpos addresses"""
        joint_map = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_map[joint_name] = self.model.jnt_qposadr[i]
        return joint_map
    
    def calibrate(self, calibration_data):
        """Calibrate neutral pose from T-pose data"""
        if calibration_data and "pose24" in calibration_data:
            self.neutral_pose = calibration_data["pose24"]
            self.is_calibrated = True
            print("✅ Calibration complete")
            return True
        return False
    
    def quat_to_euler(self, quat):
        """Convert quaternion [x,y,z,w] to Euler angles with coordinate remapping"""
        try:
            x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            
            # Normalize
            norm = np.sqrt(x*x + y*y + z*z + w*w)
            if norm < 1e-6:
                return [0.0, 0.0, 0.0]
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
            
            # Convert to Euler (XYZ order - Unity convention)
            rot = R.from_quat([x, y, z, w])
            unity_euler = rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw] in Unity
            
            # Apply coordinate remapping if enabled
            if COORDINATE_REMAP['use_remapping']:
                axis_map = COORDINATE_REMAP['axis_map']
                axis_signs = COORDINATE_REMAP['axis_signs']
                
                # Remap axes: Unity -> MuJoCo
                remapped = [0.0, 0.0, 0.0]
                for unity_axis, mujoco_axis in axis_map.items():
                    remapped[mujoco_axis] = unity_euler[unity_axis] * axis_signs[unity_axis]
                
                return remapped
            else:
                return unity_euler.tolist()
            
        except Exception as e:
            return [0.0, 0.0, 0.0]
    
    def compute_relative_rotation(self, current_quat, neutral_quat):
        """Compute relative rotation from neutral pose"""
        try:
            curr = np.array([float(q) for q in current_quat])
            neut = np.array([float(q) for q in neutral_quat])
            
            # Normalize
            curr = curr / (np.linalg.norm(curr) + 1e-8)
            neut = neut / (np.linalg.norm(neut) + 1e-8)
            
            # Relative quaternion: q_rel = q_current * q_neutral^(-1)
            neut_conj = np.array([-neut[0], -neut[1], -neut[2], neut[3]])
            
            # Quaternion multiplication
            x1, y1, z1, w1 = curr
            x2, y2, z2, w2 = neut_conj
            rel_quat = np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])
            
            return rel_quat.tolist()
        except:
            return [0, 0, 0, 1]
    
    def convert_to_qpos(self, mocap_data, current_qpos):
        """Convert ReboCap data to MuJoCo qpos"""
        qpos = current_qpos.copy()
        
        if not mocap_data or "pose24" not in mocap_data:
            return qpos
        
        pose24 = mocap_data["pose24"]
        if len(pose24) != 24:
            return qpos
        
        # Update root position if available
        if "tran" in mocap_data:
            tran = mocap_data["tran"]
            qpos[0] = tran[0]
            qpos[1] = tran[1]
            qpos[2] = tran[2] + 1.0  # Height offset
        
        # Update root orientation
        if self.is_calibrated:
            root_quat = pose24[0]
            neutral_root = self.neutral_pose[0]
            rel_quat = self.compute_relative_rotation(root_quat, neutral_root)
            qpos[3] = rel_quat[3]  # w
            qpos[4] = rel_quat[0]  # x
            qpos[5] = rel_quat[1]  # y
            qpos[6] = rel_quat[2]  # z
        
        # Process joint angles
        for rebocap_idx, joint_configs in REBOCAP_TO_G1_MAPPING.items():
            if rebocap_idx >= len(pose24):
                continue
            
            current_quat = pose24[rebocap_idx]
            
            # Get relative rotation if calibrated
            if self.is_calibrated and rebocap_idx < len(self.neutral_pose):
                neutral_quat = self.neutral_pose[rebocap_idx]
                rel_quat = self.compute_relative_rotation(current_quat, neutral_quat)
                euler = self.quat_to_euler(rel_quat)
            else:
                euler = self.quat_to_euler(current_quat)
            
            # Apply to G1 joints
            for joint_name, axis_idx, scale, offset in joint_configs:
                if joint_name in self.joint_addr_map:
                    addr = self.joint_addr_map[joint_name]
                    angle = euler[axis_idx] * scale + offset
                    qpos[addr] = angle
        
        return qpos

# ============================================
# MAIN VIEWER
# ============================================
def run_realtime_viewer(args):
    """Main real-time viewer loop"""
    
    print("\n" + "="*70)
    print("🤖 REBOCAP REAL-TIME MUJOCO G1 VIEWER")
    print("="*70)
    
    # Load model
    print(f"📂 Loading model: {args.model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(args.model_path)
        data = mujoco.MjData(model)
        print(f"✅ Model loaded: {model.nq} DOF, {model.njnt} joints")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize motion converter
    converter = MotionConverter(model)
    
    # Start receiver
    receiver = RebocapDirectReceiver(args.port)
    if not receiver.start():
        return
    
    # Wait for first data
    print("\n⏳ Waiting for first motion data...")
    timeout = 10
    start = time.time()
    first_data = None
    while not first_data and (time.time() - start) < timeout:
        first_data = receiver.get_latest_data()
        time.sleep(0.1)
    
    if not first_data:
        print("❌ No data received. Check ReboCap is running and sending data.")
        receiver.stop()
        return
    
    print("✅ Receiving motion data!")
    
    # Calibration phase
    if args.calibrate:
        print("\n" + "="*70)
        print("📏 CALIBRATION MODE")
        print("="*70)
        print("1. Stand in T-pose with arms horizontal")
        print("2. Face forward and remain still")
        print("3. Press ENTER when ready...")
        input()
        
        print("⏳ Capturing neutral pose (5 seconds)...")
        time.sleep(7)
        calibration_data = receiver.get_latest_data()
        if calibration_data:
            converter.calibrate(calibration_data)
        else:
            print("⚠️  No calibration data received, using default")
    
    print("\n" + "="*70)
    print("🎬 STARTING REAL-TIME VISUALIZATION")
    print("="*70)
    print("Controls:")
    print("   - Mouse: Rotate view")
    print("   - Scroll: Zoom")
    print("   - Double-click: Reset view")
    print("   - ESC or close window: Exit")
    print("="*70 + "\n")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Set initial camera
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.8]
        
        last_update = time.time()
        frame_count = 0
        last_fps_display = time.time()
        fps_counter = 0
        
        try:
            while viewer.is_running():
                current_time = time.time()
                
                # Get latest motion data
                motion_data = receiver.get_latest_data()
                
                if motion_data:
                    # Convert to qpos
                    new_qpos = converter.convert_to_qpos(motion_data, data.qpos)
                    data.qpos[:] = new_qpos
                    data.qvel[:] = 0
                    
                    frame_count += 1
                    fps_counter += 1
                    last_update = current_time
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
                
                # Update viewer
                viewer.sync()
                
                # FPS display
                if current_time - last_fps_display >= 2.0:
                    render_fps = fps_counter / (current_time - last_fps_display)
                    print(f"🎬 Rendering: {render_fps:.1f} FPS | "
                          f"Frames: {frame_count} | "
                          f"Connection: {'🟢' if receiver.connected else '🔴'}")
                    fps_counter = 0
                    last_fps_display = current_time
                
                # Check for stale data
                if current_time - last_update > 2.0 and receiver.connected:
                    print("⚠️  No new data received for 2 seconds")
                    last_update = current_time
                
                # Control frame rate
                time.sleep(0.01)  # ~100 Hz max
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        finally:
            receiver.stop()
            print("✅ Viewer closed")

# ============================================
# ENTRY POINT
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Real-time ReboCap MuJoCo G1 Viewer (Direct SDK)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default port 7690)
  python realtime_mujoco_viewer.py
  
  # With calibration
  python realtime_mujoco_viewer.py --calibrate
  
  # Custom port
  python realtime_mujoco_viewer.py --port 7691
  
  # Custom model path
  python realtime_mujoco_viewer.py --model_path scene.xml
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="g1_29dof.xml",
        help="Path to G1 MuJoCo XML model (default: g1_29dof.xml)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7690,
        help="ReboCap SDK port (default: 7690)"
    )
    
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration mode for T-pose neutral"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        print("\nMake sure you have:")
        print("  1. g1_29dof.xml file in current directory")
        print("  2. meshes/ folder with G1 STL files")
        return
    
    # Run viewer
    run_realtime_viewer(args)

if __name__ == "__main__":
    main()