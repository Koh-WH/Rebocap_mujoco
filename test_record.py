#!/usr/bin/env python3
"""
Recorder for ReboCap MuJoCo G1 - Saves motion to HDF5
"""

import argparse
import queue
import time
import numpy as np
import mujoco
import mujoco.viewer
import h5py
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys

# Import configuration (Expects config.py in same folder)
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
    print("❌ config.py not found! Ensure it is in the same directory.")
    sys.exit(1)

# Import ReboCap SDK
try:
    import rebocap_ws_sdk
    REBOCAP_SDK_AVAILABLE = True
except ImportError:
    print("❌ ReboCap SDK not found! Ensure 'rebocap_ws_sdk' folder is present.")
    sys.exit(1)

# ============================================
# HELPER FUNCTIONS & CLASSES (Unchanged)
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
        
        if time.time() - self.last_fps_time >= 3.0:
            # Reduced print frequency to keep console clean for recording status
            self.fps_counter = 0
            self.last_fps_time = time.time()

    def start(self):
        print(f"🔌 Connecting to ReboCap on port {self.port}...")
        res = self.sdk.open(self.port)
        self.connected = (res == 0)
        return self.connected

    def get_latest_data(self):
        try: return self.data_queue.get_nowait()
        except: return None
    
    def stop(self):
        try: self.sdk.close()
        except: pass

class MotionConverter:
    def __init__(self, model):
        self.model = model
        self.joint_map = {}
        for i in range(model.njnt):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n: 
                self.joint_map[n] = {'qpos_idx': model.jnt_qposadr[i], 'joint_id': i}
        self.neutral_pose = None
        self.is_calibrated = False

    def calibrate(self, buffer_data):
        if not buffer_data: return False
        count = len(buffer_data)
        print(f"📊 Calibrating with {count} frames...")
        sum_pose = np.zeros((24, 4))
        for frame in buffer_data:
            sum_pose += np.array(frame['pose24'])
        avg_pose = sum_pose / count
        self.neutral_pose = [normalize(q).tolist() for q in avg_pose]
        self.is_calibrated = True
        return True

    def quat_to_euler(self, q):
        try:
            x,y,z,w = normalize(np.array(q))
            rot = R.from_quat([x,y,z,w])
            eu = rot.as_euler('xyz', degrees=False)
            if COORDINATE_REMAP['use_remapping']:
                axis_map = COORDINATE_REMAP['axis_map']
                axis_signs = COORDINATE_REMAP['axis_signs']
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

    def apply_ground_clamping(self, data, foot_offset=0.045):
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
        
        t = mocap_data['tran']
        qpos[0], qpos[1], qpos[2] = t[0], t[1], t[2]

        if self.is_calibrated:
            root_quat = pose24[0]
            neutral_root = self.neutral_pose[0]
            rel_quat = self.compute_relative_rotation(root_quat, neutral_root)
            qpos[3], qpos[4], qpos[5], qpos[6] = rel_quat[3], rel_quat[0], rel_quat[1], rel_quat[2]
        
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
                        info = self.joint_map[name]
                        val = eu[axis] * scale + offset
                        if self.model.jnt_limited[info['joint_id']]:
                            min_l, max_l = self.model.jnt_range[info['joint_id']]
                            val = np.clip(val, min_l, max_l)
                        qpos[info['qpos_idx']] = val
        return qpos

# ============================================
# RECORDING FUNCTIONS
# ============================================
def save_recording(filename, buffer):
    if not buffer:
        print("\n⚠️ No data to save.")
        return

    print(f"\n💾 Saving {len(buffer)} frames to {filename}...")
    try:
        data_np = np.array(buffer)
        
        with h5py.File(filename, 'w') as f:
            # Create a structure similar to common datasets
            grp = f.create_group("data")
            demo = grp.create_group("demo_0")
            
            # Save raw observation/states
            # Assuming standard MuJoCo qpos structure: [root_pos(3), root_quat(4), joints(N)]
            obs = demo.create_group("obs")
            obs.create_dataset("robot_joint_pos", data=data_np[:, 7:]) # Joints only
            obs.create_dataset("robot_root_pos", data=data_np[:, :3])  # Root Pos
            obs.create_dataset("robot_root_rot", data=data_np[:, 3:7]) # Root Rot (w,x,y,z)
            
            # Also save full qpos for easy replay
            demo.create_dataset("qpos", data=data_np)
            
            f.attrs["total_frames"] = len(buffer)
            
        print("✅ Save complete!")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description="ReboCap G1 Recorder")
    parser.add_argument("--output", type=str, default="recordings/mocap_recording.hdf5", help="Output HDF5 filename")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--port", type=int, default=REBOCAP_PORT)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    # Setup
    if not Path(args.model_path).exists():
        print("❌ Model not found.")
        return
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    conv = MotionConverter(model)
    rx = RebocapDirectReceiver(args.port)
    
    if not rx.start(): return

    # Wait for Data
    print("⏳ Waiting for stream...")
    while not rx.get_latest_data():
        time.sleep(0.1)

    # Calibration
    if args.calibrate:
        print("STAND IN T-POSE AND PRESS ENTER")
        input()
        print("Calibrating...")
        time.sleep(CALIBRATION_WAIT_TIME)
        buf = []
        end = time.time() + 5
        while time.time() < end:
            d = rx.get_latest_data()
            if d: buf.append(d)
            time.sleep(0.05)
        conv.calibrate(buf)
        input("Press ENTER to start recording...")

    # Recording Loop
    recording_buffer = []
    print(f"\n🔴 RECORDING STARTED -> {args.output}")
    print("Press Ctrl+C to stop and save.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        
        try:
            while viewer.is_running():
                d = rx.get_latest_data()
                if d:
                    # Update Robot
                    data.qpos[:] = conv.convert_to_qpos(d, data.qpos)
                    data.qvel[:] = 0
                    conv.apply_ground_clamping(data)
                    
                    # RECORD FRAME
                    recording_buffer.append(data.qpos.copy())
                
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01) # ~100Hz cap
                
        except KeyboardInterrupt:
            print("\n⏹️  Stop signal received.")
        finally:
            rx.stop()
            save_recording(args.output, recording_buffer)

if __name__ == "__main__":
    main()