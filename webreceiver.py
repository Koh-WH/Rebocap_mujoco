#!/usr/bin/env python3
"""
WEBRECEIVER: Remote G1 Viewer
- Connects to the Ngrok URL
- Renders robot with Locked Hips & Ground Clamping
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
    from config import COORDINATE_REMAP, REBOCAP_TO_G1_MAPPING
except ImportError:
    print("❌ config.py not found! Copy it from your host PC.")
    sys.exit(1)

# --- MATH HELPERS ---
def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n

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

# --- WEBSOCKET CLIENT ---
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

# --- CONVERTER (Locked Hips & Clamping) ---
class MotionConverter:
    def __init__(self, model):
        self.model = model
        self.joint_map = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i] for i in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}
        self.neutral_pose = None
        self.is_calibrated = False

    def calibrate(self, buffer_data):
        if not buffer_data: return False
        sum_pose = np.zeros((24, 4))
        for frame in buffer_data:
            sum_pose += np.array(frame['pose24'])
        self.neutral_pose = [normalize(q).tolist() for q in (sum_pose / len(buffer_data))]
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
        
        # 1. Root Position (Start at 0 for clamping)
        t = mocap_data['tran']
        qpos[0] = t[0]
        qpos[1] = t[1]
        qpos[2] = t[2] + 0.0 

        # 2. Root Rotation -> LOCKED UPRIGHT
        qpos[3], qpos[4], qpos[5], qpos[6] = 1.0, 0.0, 0.0, 0.0
        
        # 3. Joints (Mapped from Config)
        pose24 = mocap_data['pose24']
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
                        val = eu[axis] * scale + offset
                        qpos[self.joint_map[name]] = val
                    
        return qpos

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="Ngrok URL")
    parser.add_argument("--model_path", default="scene.xml")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if not args.url:
        print("\n" + "="*60)
        args.url = input("🔗 Enter Ngrok URL (from webserver_ngrok.py): ").strip()
        print("="*60 + "\n")
        
    # Auto-fix protocol for Ngrok
    if "http" in args.url: args.url = args.url.replace("http", "ws")
    if not args.url.startswith("ws"): args.url = "wss://" + args.url

    if not Path(args.model_path).exists():
        print(f"❌ File not found: {args.model_path}")
        return

    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    conv = MotionConverter(model)
    rx = RemoteReceiver()
    
    rx.connect(args.url)

    print("⏳ Waiting for stream...")
    while not rx.get_latest(): time.sleep(0.1)
    print("✅ Connected!")

    if args.calibrate:
        print("\n📏 CALIBRATION: Press ENTER in T-POSE...")
        input()
        print("📸 Capturing 5s...")
        buf = []
        end = time.time()+5
        while time.time()<end:
            d = rx.get_latest()
            if d: buf.append(d)
            time.sleep(0.05)
        conv.calibrate(buf)
        print("✅ Calibrated!")

    print("\n🎬 STARTING VIEWER (Press 'S' for shadows)")
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