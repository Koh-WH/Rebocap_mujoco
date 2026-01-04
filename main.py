#!/usr/bin/env python3
"""
Real-Time ReboCap MuJoCo G1 Viewer - LOCKED HIPS & GROUND CLAMPING
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
    from config import COORDINATE_REMAP, REBOCAP_TO_G1_MAPPING
except ImportError:
    print("❌ config.py not found!")
    sys.exit(1)

# Import ReboCap SDK
try:
    import rebocap_ws_sdk
    REBOCAP_SDK_AVAILABLE = True
except ImportError:
    print("❌ ReboCap SDK not found!")
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
        self.sdk = rebocap_ws_sdk.RebocapWsSdk(
            coordinate_type=rebocap_ws_sdk.CoordinateType.UnityCoordinate,
            use_global_rotation=False
        )
        self.sdk.set_pose_msg_callback(self._on_pose)

    def _on_pose(self, sdk, tran, pose24, static, ts):
        if not pose24 or len(pose24) != 24: return
        data = { "tran": list(tran) if tran else [0]*3, "pose24": [list(q) for q in pose24] }
        if self.data_queue.full():
            try: self.data_queue.get_nowait()
            except: pass
        self.data_queue.put(data)

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

# ============================================
# MOTION CONVERTER (Modified for Locking & Physics)
# ============================================
class MotionConverter:
    def __init__(self, model):
        self.model = model
        self.joint_map = {}
        for i in range(model.njnt):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n: self.joint_map[n] = model.jnt_qposadr[i]
        
        self.neutral_pose = None
        self.is_calibrated = False

    def calibrate(self, buffer_data):
        if not buffer_data: return False
        count = len(buffer_data)
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
        
        # 1. Root Position
        t = mocap_data['tran']
        qpos[0] = t[0]
        qpos[1] = t[1]
        # Start at 0.0 instead of 1.0, let clamping fix the height
        qpos[2] = t[2] + 0.0 

        # 2. Root Rotation -> FORCED LOCKED
        # Ignore ReboCap root rotation and force upright identity
        qpos[3] = 1.0 # w
        qpos[4] = 0.0 # x
        qpos[5] = 0.0 # y
        qpos[6] = 0.0 # z
        
        # 3. Joints (Uses YOUR config mapping)
        pose24 = mocap_data['pose24']
        for idx, cfg in REBOCAP_TO_G1_MAPPING.items():
            if idx >= len(pose24): continue
            
            q_curr = pose24[idx]
            if self.is_calibrated:
                # Still calculate relative rotation so limbs move correctly
                q_curr = self.compute_relative_rotation(q_curr, self.neutral_pose[idx])
                # Note: compute_relative_rotation returns list, convert if needed
            
            eu = self.quat_to_euler(q_curr)
            
            for item in cfg:
                # Handle both 4-item and 6-item (safety) tuples
                if len(item) >= 4:
                    name, axis, scale, offset = item[0], item[1], item[2], item[3]
                    if name in self.joint_map:
                        val = eu[axis] * scale + offset
                        qpos[self.joint_map[name]] = val
                    
        return qpos

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="xml/scene.xml")
    parser.add_argument("--port", type=int, default=7690)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"❌ File not found: {args.model_path}")
        return

    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    
    conv = MotionConverter(model)
    rx = RebocapDirectReceiver(args.port)
    
    if not rx.start(): return

    print("\n⏳ Waiting for data...")
    while not rx.get_latest_data(): time.sleep(0.1)
    print("✅ Data received!")

    if args.calibrate:
        print("\n📏 CALIBRATION: Stand in T-POSE and press ENTER...")
        input()
        print("📸 Capturing 5s...")
        buf = []
        end = time.time()+5
        while time.time()<end:
            d = rx.get_latest_data()
            if d: buf.append(d)
            time.sleep(0.05)
        conv.calibrate(buf)
        print("✅ Calibrated!")

    print("\n🎬 STARTING VIEWER")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.8]
        
        while viewer.is_running():
            d = rx.get_latest_data()
            if d:
                data.qpos[:] = conv.convert_to_qpos(d, data.qpos)
                data.qvel[:] = 0
                conv.apply_ground_clamping(data)
            
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    rx.stop()
    print("✅ Viewer closed")

if __name__ == "__main__":
    main()