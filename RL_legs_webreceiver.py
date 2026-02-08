#!/usr/bin/env python3
"""
TELEOP DANCE: UNIVERSAL RL LEGS CONTROLLER (GLOBAL ROTATION MODE)
- Integrated Web Receiver
- Uses config.py logic for accurate upper body tracking
"""

import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import json
import websocket
import threading
import queue
from scipy.spatial.transform import Rotation as R
import sys

# Import configuration
try:
    from config import (
        REBOCAP_TO_G1_MAPPING,
        ROOT_HEIGHT_OFFSET,
        ROOT_ROTATION_CONFIG
    )
except ImportError:
    print("❌ config.py not found!")
    print("Make sure config.py is in the same directory as this script.")
    sys.exit(1)

# --- CONFIGURATION ---
LEG_INDICES = list(range(12))            # Joints 0-11 (Legs)
UPPER_BODY_INDICES = list(range(12, 29)) # Joints 12-28 (Waist + Arms + Head)

# GAINS
LEG_KP = 100.0    
LEG_KD = 10.0     
ARM_KP = 40.0
ARM_KD = 1.0
COM_KP = [150.0, 150.0, 0.0] 
COM_KD = [10.0, 10.0, 0.0]
DECIMATION = 10  

# ============================================
# HELPER MATH FUNCTIONS (FROM MAIN.PY)
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

def compute_relative_quat(child_quat, parent_quat):
    parent_inv = quat_conj(normalize(np.array(parent_quat)))
    child_norm = normalize(np.array(child_quat))
    return quat_mul(parent_inv, child_norm)

# ============================================
# MOTION CONVERTER (LOGIC FROM MAIN.PY)
# ============================================
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
        print(f"📊 Calibrating with {len(buffer_data)} frames...")
        sum_pose = np.zeros((24, 4))
        for frame in buffer_data:
            sum_pose += np.array(frame['pose24'])
        avg_pose = sum_pose / len(buffer_data)
        self.neutral_pose = [normalize(q).tolist() for q in avg_pose]
        self.is_calibrated = True
        print("✅ Calibration complete!")
        return True

    def quat_to_euler(self, q):
        try:
            x, y, z, w = normalize(np.array(q))
            rot = R.from_quat([x, y, z, w])
            eu = rot.as_euler('xyz', degrees=False)
            return {'x': eu[0], 'y': eu[1], 'z': eu[2]}
        except: return {'x': 0, 'y': 0, 'z': 0}

    def convert_to_qpos(self, mocap_data, current_qpos):
        qpos = current_qpos.copy()
        if not mocap_data or "pose24" not in mocap_data: return qpos
        
        pose24 = mocap_data['pose24']
        
        # 1. Root Position
        t = mocap_data['tran']
        qpos[0] = t[0]
        qpos[1] = t[1]
        qpos[2] = t[2] + ROOT_HEIGHT_OFFSET

        # 2. Root Rotation
        pelvis_quat = pose24[0]
        if self.is_calibrated:
            rel_quat = compute_relative_quat(pelvis_quat, self.neutral_pose[0])
        else:
            rel_quat = normalize(np.array(pelvis_quat))
        
        euler = self.quat_to_euler(rel_quat)
        root_euler = [
            euler['x'] if ROOT_ROTATION_CONFIG['use_roll'] else 0.0,
            euler['z'] if ROOT_ROTATION_CONFIG['use_pitch'] else 0.0,
            euler['y'] if ROOT_ROTATION_CONFIG['use_yaw'] else 0.0
        ]
        root_rot = R.from_euler('xyz', root_euler, degrees=False)
        root_quat = root_rot.as_quat()
        qpos[3:7] = [root_quat[3], root_quat[0], root_quat[1], root_quat[2]] # w,x,y,z
        
        # 3. Process Joints
        for mapping in REBOCAP_TO_G1_MAPPING:
            joint_name = mapping['joint']
            if joint_name not in self.joint_map: continue
            
            child_idx = mapping['sensor']
            parent_idx = mapping.get('parent_sensor', None)
            
            # Logic from main.py
            if parent_idx is not None:
                quat_relative = compute_relative_quat(pose24[child_idx], pose24[parent_idx])
            else:
                quat_relative = pose24[child_idx]
            
            if self.is_calibrated:
                neutral = self.neutral_pose[child_idx]
                quat_relative = compute_relative_quat(quat_relative, neutral)
                
            euler = self.quat_to_euler(quat_relative)
            val = euler[mapping['axis']] * mapping.get('scale', 1.0) + mapping.get('offset', 0.0)
            
            info = self.joint_map[joint_name]
            if self.model.jnt_limited[info['joint_id']]:
                val = np.clip(val, *self.model.jnt_range[info['joint_id']])
            
            qpos[info['qpos_idx']] = val
            
        return qpos

# ============================================
# WEBSOCKET RECEIVER
# ============================================
class WebReceiver:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=2)
        self.running = False
        self.ws = None

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if self.data_queue.full():
                try: self.data_queue.get_nowait()
                except: pass
            self.data_queue.put(data)
        except: pass

    def connect(self, url):
        self.running = True
        print(f"🔌 Connecting to {url}...")
        self.ws = websocket.WebSocketApp(url, on_message=self.on_message)
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def get_latest(self):
        try: return self.data_queue.get_nowait()
        except: return None
        
    def stop(self):
        self.running = False
        if self.ws: self.ws.close()

# ============================================
# CONTROLLER
# ============================================
class G1Controller:
    def __init__(self, model_path, policy_path):
        print(f"📂 Loading XML: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"🧠 Loading RL Policy: {policy_path}")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        self.policy_dof = self._detect_policy_dof()
        print(f"🔍 Detected Policy Structure: {self.policy_dof}-DoF")
        
        self.receiver = WebReceiver()
        self.converter = MotionConverter(self.model)
        
        self.last_actions = torch.zeros(self.policy_dof)
        self.default_dof_pos = torch.zeros(29)
        
        if self.model.nkey > 0:
            self.default_dof_pos = torch.from_numpy(self.model.key_qpos[0][7:7+29]).float()
        else:
            bent_knees = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
            self.default_dof_pos[0:12] = torch.from_numpy(bent_knees)
            
        self.jac_com = np.zeros((3, self.model.nv))

    def _detect_policy_dof(self):
        for dof in [12, 23, 29]:
            try:
                self.policy(torch.zeros(1, 11 + (3 * dof)))
                return dof
            except: continue
        return 12

    def compute_balance_torque(self):
        if not hasattr(self, 'com_ref'):
            self.com_ref = self.data.subtree_com[0].copy()
            self.com_ref[2] = 0.0 

        com_pos = self.data.subtree_com[0]
        com_vel = self.data.subtree_linvel[0]
        target_pos = self.com_ref.copy()
        target_pos[2] = com_pos[2] 
        
        virtual_force = np.array([
            COM_KP[0] * (target_pos[0] - com_pos[0]) - COM_KD[0] * com_vel[0],
            COM_KP[1] * (target_pos[1] - com_pos[1]) - COM_KD[1] * com_vel[1],
            0.0
        ])
        
        mujoco.mj_jacSubtreeCom(self.model, self.data, self.jac_com, 0)
        return (self.jac_com.T @ virtual_force)[6:18]

    def get_observations(self):
        base_quat = self.data.qpos[3:7]
        base_ang_vel = torch.tensor(self.data.qvel[3:6], dtype=torch.float32) * 0.25
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        proj_grav = torch.tensor(r.inv().apply([0, 0, -1]), dtype=torch.float32)
        
        N = self.policy_dof
        current_dof_pos = torch.tensor(self.data.qpos[7:7+N], dtype=torch.float32)
        current_dof_vel = torch.tensor(self.data.qvel[6:6+N], dtype=torch.float32)
        target_dof_pos = (current_dof_pos - self.default_dof_pos[:N]) * 1.0
        
        return torch.cat([
            base_ang_vel, proj_grav, torch.zeros(3), 
            target_dof_pos, current_dof_vel * 0.05, 
            self.last_actions, torch.tensor([0.0, 1.0])
        ]).unsqueeze(0)

    def run(self, url):
        self.receiver.connect(url)
        print("⏳ Waiting for Data...")
        while not self.receiver.get_latest(): time.sleep(0.1)
        
        # --- CALIBRATION ---
        print("\n🎯 CALIBRATION NEEDED!")
        print("1. Stand in T-pose (Neutral)")
        print("2. Press ENTER to calibrate...")
        input()
        print("⏳ Calibrating...")
        
        buf = []
        end = time.time() + 3
        while time.time() < end:
            d = self.receiver.get_latest()
            if d: buf.append(d)
            time.sleep(0.05)
            
        self.converter.calibrate(buf)
        input("✅ Calibrated! Press ENTER to start control...")
        
        # --- MAIN LOOP ---
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Reset
            if self.model.nkey > 0: self.data.qpos[:] = self.model.key_qpos[0]
            else: 
                self.data.qpos[:] = 0.0
                self.data.qpos[3] = 1.0
                self.data.qpos[7:7+29] = self.default_dof_pos.numpy()
            
            self.data.qpos[2] = 0.82
            mujoco.mj_forward(self.model, self.data)
            self.com_ref = self.data.subtree_com[0].copy()

            step = 0
            rl_leg_targets = self.default_dof_pos[0:12].numpy()
            upper_body_targets = self.default_dof_pos[12:29].numpy()

            while viewer.is_running():
                start = time.time()
                
                # 1. RL Step
                if step % DECIMATION == 0:
                    with torch.no_grad():
                        obs = self.get_observations()
                        actions = self.policy(obs).detach()[0]
                        self.last_actions = actions
                        rl_leg_targets = (actions[0:12].numpy() * 0.5) + self.default_dof_pos[0:12].numpy()

                # 2. Mocap Update (Upper Body)
                mocap_data = self.receiver.get_latest()
                if mocap_data:
                    # Convert using global-relative logic
                    full_qpos = self.converter.convert_to_qpos(mocap_data, self.data.qpos)
                    upper_body_targets = full_qpos[7+12 : 7+29]

                # 3. Control
                qpos_legs = self.data.qpos[7 : 7+12]
                qvel_legs = self.data.qvel[6 : 6+12]
                leg_torques = LEG_KP * (rl_leg_targets - qpos_legs) - LEG_KD * qvel_legs + self.compute_balance_torque()

                qpos_arms = self.data.qpos[7+12 : 7+29]
                qvel_arms = self.data.qvel[6+12 : 6+29]
                arm_torques = ARM_KP * (upper_body_targets - qpos_arms) - ARM_KD * qvel_arms

                self.data.ctrl[LEG_INDICES] = np.clip(leg_torques, -100, 100)
                self.data.ctrl[UPPER_BODY_INDICES] = np.clip(arm_torques, -60, 60)
                
                mujoco.mj_step(self.model, self.data)
                if step % 10 == 0: viewer.sync()
                step += 1
                
                elapsed = time.time() - start
                if elapsed < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - elapsed)
        self.receiver.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="wss://improvable-maile-unquerulously.ngrok-free.dev")
    parser.add_argument("--policy", type=str, required=True, help="Path to .pt policy")
    parser.add_argument("--xml", type=str, default="xml/scene.xml")
    args = parser.parse_args()
    
    sim = G1Controller(args.xml, args.policy)
    sim.run(args.url)