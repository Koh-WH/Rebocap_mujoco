#!/usr/bin/env python3
"""
WEBRECEIVER: Remote G1 Viewer (Global Rotation Mode)
- Connects to the Ngrok URL
- Uses "Global Rotation" logic to match main.py
"""
import argparse
import asyncio
import json
import threading
import time
import numpy as np
import mujoco
import mujoco.viewer
import websockets
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys

# Import Config
try:
    # We now import the NEW config variables
    from config import (
        REBOCAP_TO_G1_MAPPING,
        ROOT_HEIGHT_OFFSET,
        ROOT_ROTATION_CONFIG
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure 'config.py' is the renamed 'config_global.py' and is in this folder.")
    sys.exit(1)

# ============================================
# HELPER MATH FUNCTIONS
# ============================================
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm < 1e-8 else v / norm

def quat_mul(q1, q2):
    """Multiply two quaternions [x, y, z, w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_conj(q):
    """Conjugate of quaternion [x, y, z, w]"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def compute_relative_quat(child_quat, parent_quat):
    """Compute relative rotation: child relative to parent"""
    parent_inv = quat_conj(normalize(np.array(parent_quat)))
    child_norm = normalize(np.array(child_quat))
    return quat_mul(parent_inv, child_norm)

# ============================================
# WEBSOCKET CLIENT
# ============================================
class RemoteReceiver:
    def __init__(self):
        self.latest_data = None
        self.running = False
        self.connected = False

    def connect(self, url):
        self.url = url
        self.running = True
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._listen())

    async def _listen(self):
        while self.running:
            try:
                print(f"🔄 Connecting to {self.url}...")
                async with websockets.connect(self.url) as ws:
                    print("✅ Connected to Ngrok Tunnel!")
                    self.connected = True
                    async for msg in ws:
                        if not self.running: break
                        try: self.latest_data = json.loads(msg)
                        except: pass
            except Exception as e:
                self.connected = False
                print(f"⚠️ Connection lost: {e}. Retrying in 3s...")
                await asyncio.sleep(3)

    def get_latest(self):
        return self.latest_data

    def stop(self):
        self.running = False

# ============================================
# MOTION CONVERTER (NEW GLOBAL LOGIC)
# ============================================
class MotionConverter:
    def __init__(self, model):
        self.model = model
        self.joint_map = {}
        # Store both qpos index and joint ID
        for i in range(model.njnt):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n: 
                self.joint_map[n] = {
                    'qpos_idx': model.jnt_qposadr[i],
                    'joint_id': i
                }
        
        self.neutral_pose = None
        self.is_calibrated = False

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
        
        print(f"✅ Calibration complete!")
        return True

    def quat_to_euler(self, q):
        try:
            x, y, z, w = normalize(np.array(q))
            rot = R.from_quat([x, y, z, w])
            eu = rot.as_euler('xyz', degrees=False)
            return {'x': eu[0], 'y': eu[1], 'z': eu[2]}
        except: 
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

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
        qpos[2] = t[2] + ROOT_HEIGHT_OFFSET

        # 2. Root Rotation (Pelvis - index 0)
        pelvis_quat = pose24[0]
        
        if self.is_calibrated:
            neutral_pelvis = self.neutral_pose[0]
            rel_quat = compute_relative_quat(pelvis_quat, neutral_pelvis)
        else:
            rel_quat = normalize(np.array(pelvis_quat))
        
        euler = self.quat_to_euler(rel_quat)
        
        root_euler = [
            euler['x'] if ROOT_ROTATION_CONFIG['use_roll'] else 0.0,
            euler['z'] if ROOT_ROTATION_CONFIG['use_pitch'] else 0.0,
            euler['y'] if ROOT_ROTATION_CONFIG['use_yaw'] else 0.0
        ]
        
        from scipy.spatial.transform import Rotation as Rot
        root_rot = Rot.from_euler('xyz', root_euler, degrees=False)
        root_quat = root_rot.as_quat()  # [x, y, z, w]
        
        qpos[3] = root_quat[3]   # w
        qpos[4] = root_quat[0]   # x
        qpos[5] = root_quat[1]   # y
        qpos[6] = root_quat[2]   # z
        
        # 3. Process Joint Mappings
        for mapping in REBOCAP_TO_G1_MAPPING:
            joint_name = mapping['joint']
            child_idx = mapping['sensor']
            parent_idx = mapping.get('parent_sensor', None)
            axis = mapping['axis']
            scale = mapping.get('scale', 1.0)
            offset = mapping.get('offset', 0.0)
            
            if child_idx >= len(pose24): continue
            if joint_name not in self.joint_map: continue
            
            # Get sensor quaternions
            child_quat_global = pose24[child_idx]
            
            # Compute relative rotation
            if parent_idx is not None and parent_idx < len(pose24):
                parent_quat_global = pose24[parent_idx]
                quat_relative = compute_relative_quat(child_quat_global, parent_quat_global)
            else:
                quat_relative = child_quat_global
            
            # Apply calibration
            if self.is_calibrated and child_idx < len(self.neutral_pose):
                neutral_quat = self.neutral_pose[child_idx]
                quat_relative = compute_relative_quat(quat_relative, neutral_quat)
            
            # Convert to Euler and extract axis
            euler = self.quat_to_euler(quat_relative)
            val = euler[axis] * scale + offset
            
            # Apply Limits
            info = self.joint_map[joint_name]
            q_idx = info['qpos_idx']
            j_id = info['joint_id']
            
            if self.model.jnt_limited[j_id]:
                min_limit, max_limit = self.model.jnt_range[j_id]
                val = np.clip(val, min_limit, max_limit)
            
            qpos[q_idx] = val
        
        return qpos

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="wss://improvable-maile-unquerulously.ngrok-free.dev", help="Ngrok URL")
    parser.add_argument("--model_path", default="xml/scene.xml")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if not args.url:
        print("\n" + "="*60)
        args.url = input("🔗 Enter Ngrok URL (from webserver.py): ").strip()
        print("="*60 + "\n")
        
    # Auto-fix protocol
    if "http" in args.url: args.url = args.url.replace("http", "ws")
    if not args.url.startswith("ws"): args.url = "wss://" + args.url

    if not Path(args.model_path).exists():
        print(f"❌ File not found: {args.model_path}")
        return

    # Load Model
    print(f"\n📂 Loading model: {args.model_path}")
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    conv = MotionConverter(model)
    rx = RemoteReceiver()
    
    rx.connect(args.url)

    print("⏳ Waiting for stream...")
    while not rx.get_latest(): time.sleep(0.1)
    print("✅ Receiving data!")

    # Calibration Step
    if args.calibrate:
        print("\n" + "="*40)
        print("🎯 CALIBRATION MODE")
        print("1. Stand in T-pose")
        print("2. Press ENTER to start...")
        input()
        print("⏳ Capturing 3 seconds...")
        
        buf = []
        end = time.time() + 3
        while time.time() < end:
            d = rx.get_latest()
            if d: buf.append(d)
            time.sleep(0.05)
        
        conv.calibrate(buf)
        print("✅ Calibrated! Starting viewer...")

    # Viewer Loop
    print("\n🎬 STARTING REAL-TIME VIEWER")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.8]
        
        while viewer.is_running():
            d = rx.get_latest()
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