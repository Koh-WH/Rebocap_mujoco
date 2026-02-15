#!/usr/bin/env python3
"""
Real-Time ReboCap MuJoCo G1 Viewer - GROUND CLAMPING + SENSOR HIP TRACKING + JOINT LIMITS
"""

import argparse
import queue
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys

# Import configuration
try:
    from config import (
        COORDINATE_REMAP,
        REBOCAP_TO_G1_MAPPING,
        MODEL_PATH,
        REBOCAP_PORT,
        ROOT_HEIGHT_OFFSET,
        CALIBRATION_WAIT_TIME
    )
except ImportError:
    print("❌ config.py not found!")
    print("Make sure config.py is in the same directory")
    sys.exit(1)

# Import ReboCap SDK
try:
    import rebocap_ws_sdk
    REBOCAP_SDK_AVAILABLE = True
except ImportError:
    print("❌ ReboCap SDK not found!")
    print("Make sure 'rebocap_ws_sdk' folder is in the same directory")
    sys.exit(1)

# ============================================
# HELPER FUNCTIONS
# ============================================
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm < 1e-8 else v / norm

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

# ============================================
# RECEIVER CLASS 
# ============================================
class RebocapDirectReceiver:
    def __init__(self, port=7690):
        self.port = port
        self.data_queue = queue.Queue(maxsize=5)
        self.connected = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        self.sdk = rebocap_ws_sdk.RebocapWsSdk(
            coordinate_type=rebocap_ws_sdk.CoordinateType.UnityCoordinate,
            use_global_rotation=False
        )
        self.sdk.set_pose_msg_callback(self._on_pose)

    def _on_pose(self, sdk, tran, pose24, static, ts):
        if not pose24 or len(pose24) != 24: return
        data = { "tran": list(tran) if tran else [0]*3, "pose24": [list(q) for q in pose24] }
        
        self.frame_count += 1
        self.fps_counter += 1
        
        if self.data_queue.full():
            try: self.data_queue.get_nowait()
            except: pass
        self.data_queue.put(data)
        
        # FPS display every 3 seconds
        if time.time() - self.last_fps_time >= 3.0:
            fps = self.fps_counter / (time.time() - self.last_fps_time)
            print(f"📊 Receiving: {fps:.1f} FPS | Total frames: {self.frame_count}")
            self.fps_counter = 0
            self.last_fps_time = time.time()

    def start(self):
        print(f"🔌 Connecting to ReboCap on port {self.port}...")
        res = self.sdk.open(self.port)
        self.connected = (res == 0)
        if self.connected:
            print("✅ Connected to ReboCap SDK!")
            print("📡 Waiting for motion data...")
        else:
            print(f"❌ Failed to connect (error code: {res})")
            print("\nTroubleshooting:")
            print("  1. Make sure ReboCap software is running")
            print(f"  2. Check that ReboCap is broadcasting on port {self.port}")
            print("  3. Verify no other application is using the port")
        return self.connected

    def get_latest_data(self):
        try: return self.data_queue.get_nowait()
        except: return None
    
    def stop(self):
        try: 
            self.sdk.close()
            print("🔌 ReboCap SDK closed")
        except: pass

# ============================================
# MOTION CONVERTER (Modified for JOINT LIMITS)
# ============================================
class MotionConverter:
    def __init__(self, model):
        self.model = model
        self.joint_map = {}
        
        # Store both qpos index and joint ID (needed to lookup limits)
        for i in range(model.njnt):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n: 
                self.joint_map[n] = {
                    'qpos_idx': model.jnt_qposadr[i],
                    'joint_id': i
                }
        
        self.neutral_pose = None
        self.is_calibrated = False
        
        print(f"📋 Joint mapping initialized:")
        print(f"   - G1 joints found: {len(self.joint_map)}")
        print(f"   - ReboCap joints mapped: {len(REBOCAP_TO_G1_MAPPING)}")

    def calibrate(self, buffer_data):
        if not buffer_data: return False
        count = len(buffer_data)
        print(f"📊 Processing {count} calibration frames...")
        
        sum_pose = np.zeros((24, 4))
        for frame in buffer_data:
            sum_pose += np.array(frame['pose24'])
        avg_pose = sum_pose / count
        self.neutral_pose = [normalize(q).tolist() for q in avg_pose]
        self.is_calibrated = True
        
        print(f"✅ Calibration complete with {count} frames!")
        return True

    def quat_to_euler(self, q):
        try:
            x,y,z,w = normalize(np.array(q))
            rot = R.from_quat([x,y,z,w])
            eu = rot.as_euler('xyz', degrees=False)
            if COORDINATE_REMAP['use_remapping']:
                axis_map = COORDINATE_REMAP['axis_map']
                axis_signs = COORDINATE_REMAP['axis_signs']
                
                # Remap axes according to config
                remapped = [0.0, 0.0, 0.0]
                for u_ax, m_ax in axis_map.items():
                    remapped[m_ax] = eu[u_ax] * axis_signs[u_ax]
                return remapped
            return eu
        except: return [0,0,0]

    def compute_relative_rotation(self, current_quat, neutral_quat):
        try:
            curr = normalize(np.array(current_quat))
            neut = normalize(np.array(neutral_quat))
            return quat_mul(curr, quat_conj(neut)).tolist()
        except: return [0,0,0,1]

    # --- Ground Clamping Logic ---
    def apply_ground_clamping(self, data, foot_offset=0.045):
        """Keep robot feet above ground"""
        mujoco.mj_kinematics(self.model, data)
        try:
            l_z = data.body('left_ankle_roll_link').xpos[2]
            r_z = data.body('right_ankle_roll_link').xpos[2]
            lowest = min(l_z, r_z)
            if lowest < foot_offset:
                data.qpos[2] += (foot_offset - lowest)
        except: pass

    def convert_to_qpos(self, mocap_data, current_qpos):
        qpos = current_qpos.copy()
        if not mocap_data or "pose24" not in mocap_data: return qpos
        
        pose24 = mocap_data['pose24']
        
        # 1. Root Position
        t = mocap_data['tran']
        qpos[0] = t[0]
        qpos[1] = t[1]
        qpos[2] = t[2] + 0.0 

        # 2. Root Rotation
        if self.is_calibrated:
            root_quat = pose24[0]
            neutral_root = self.neutral_pose[0]
            rel_quat = self.compute_relative_rotation(root_quat, neutral_root)
            qpos[3] = rel_quat[3]  # w
            qpos[4] = rel_quat[0]  # x
            qpos[5] = rel_quat[1]  # y
            qpos[6] = rel_quat[2]  # z
        
        # 3. Joints (Uses config.py REBOCAP_TO_G1_MAPPING)
        for idx, cfg in REBOCAP_TO_G1_MAPPING.items():
            if idx >= len(pose24): continue
            
            q_curr = pose24[idx]
            if self.is_calibrated:
                q_curr = self.compute_relative_rotation(q_curr, self.neutral_pose[idx])
            
            eu = self.quat_to_euler(q_curr)
            
            for item in cfg:
                if len(item) >= 4:
                    name, axis, scale, offset = item[0], item[1], item[2], item[3]
                    
                    if name in self.joint_map:
                        # Get Joint Info
                        info = self.joint_map[name]
                        q_idx = info['qpos_idx']
                        j_id = info['joint_id']
                        
                        # Calculate raw value
                        val = eu[axis] * scale + offset
                        
                        # --- APPLY JOINT LIMITS ---
                        # Check if limits are enabled for this joint (jnt_limited == 1)
                        if self.model.jnt_limited[j_id]:
                            # Get [min, max] range from model
                            min_limit, max_limit = self.model.jnt_range[j_id]
                            # Clip value
                            val = np.clip(val, min_limit, max_limit)
                        
                        # Assign safe value
                        qpos[q_idx] = val
                    
        return qpos

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Real-time ReboCap MuJoCo G1 Viewer with Joint Limits",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to MuJoCo model (default from config.py: {MODEL_PATH})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=REBOCAP_PORT,
        help=f"ReboCap SDK port (default from config.py: {REBOCAP_PORT})"
    )
    
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration mode for T-pose neutral"
    )
    
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🤖 REBOCAP REAL-TIME MUJOCO G1 VIEWER")
    print("="*70)

    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        return

    # Load model
    print(f"📂 Loading model: {args.model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(args.model_path)
        data = mujoco.MjData(model)
        print(f"✅ Model loaded: {model.nq} DOF, {model.njnt} joints")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize converter and receiver
    conv = MotionConverter(model)
    rx = RebocapDirectReceiver(args.port)
    
    if not rx.start():
        return

    # Wait for first data
    print("\n⏳ Waiting for first motion data...")
    timeout = 10
    start_time = time.time()
    first_data = None
    while not first_data and (time.time() - start_time) < timeout:
        first_data = rx.get_latest_data()
        time.sleep(0.1)
    
    if not first_data:
        print("❌ No data received.")
        rx.stop()
        return
    
    print("✅ Receiving motion data!")

    if args.calibrate:
        print("\n" + "="*70)
        print("📏 CALIBRATION MODE")
        print("1. Stand in T-pose")
        print("2. Press ENTER...")
        input()
        print(f"⏳ Capturing...")
        time.sleep(CALIBRATION_WAIT_TIME)
        
        buf = []
        end = time.time() + 5
        while time.time() < end:
            d = rx.get_latest_data()
            if d: buf.append(d)
            time.sleep(0.05)
        
        conv.calibrate(buf)
        input("Press ENTER to start...")

    # Launch viewer
    print("\n" + "="*70)
    print("🎬 STARTING REAL-TIME VISUALIZATION")
    print("   (Joint limits enabled)")
    print("="*70 + "\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
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
                d = rx.get_latest_data()
                if d:
                    data.qpos[:] = conv.convert_to_qpos(d, data.qpos)
                    data.qvel[:] = 0
                    conv.apply_ground_clamping(data)
                    
                    frame_count += 1
                    fps_counter += 1
                    last_update = current_time
                
                mujoco.mj_forward(model, data)
                viewer.sync()
                
                if current_time - last_fps_display >= 2.0:
                    render_fps = fps_counter / (current_time - last_fps_display)
                    print(f"🎬 Rendering: {render_fps:.1f} FPS", end='\r')
                    fps_counter = 0
                    last_fps_display = current_time
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        finally:
            rx.stop()

if __name__ == "__main__":
    main()