#!/usr/bin/env python3
"""
TELEOP DANCE: PURE RL LEGS
- LEGS: 100% Controlled by AI Balance Policy (No Teleop)
- ARMS/TORSO: Controlled by VR/Mocap
- INCLUDES: CoM Stabilizer to handle the weight shifts from arm movements
"""

import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from webreceiver import RemoteReceiver, MotionConverter
from scipy.spatial.transform import Rotation as R
import sys

# --- CONFIGURATION ---
LEG_INDICES = list(range(12))        # RL Controls these
UPPER_BODY_INDICES = list(range(12, 29)) # You Control these

# 1. GAINS (Stiff legs for balance, Soft arms for safety)
LEG_KP = 100.0    
LEG_KD = 10.0     
ARM_KP = 40.0
ARM_KD = 1.0

# 2. BALANCE ARBITRATION (Helps RL handle arm swings)
# Pushes legs to correct Center of Mass shifts
COM_KP = [150.0, 150.0, 0.0] 
COM_KD = [10.0, 10.0, 0.0]

# 3. TIMING (50Hz Policy / 500Hz Sim)
DECIMATION = 10  

class G1Controller:
    def __init__(self, model_path, policy_path):
        print(f"📂 Loading XML: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"🧠 Loading RL Policy: {policy_path}")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        self.receiver = RemoteReceiver()
        self.converter = MotionConverter(self.model)
        
        self.last_actions = torch.zeros(12)
        
        # --- ROBUST DEFAULT POSE (Bent Knees) ---
        if self.model.nkey > 0:
            self.default_dof_pos = torch.from_numpy(self.model.key_qpos[0][7:7+12]).float()
        else:
            bent_knees = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0
            ], dtype=np.float32)
            self.default_dof_pos = torch.from_numpy(bent_knees)
            
        # Physics Cache
        self.jac_com = np.zeros((3, self.model.nv))

    def get_com_state(self):
        com_pos = self.data.subtree_com[0]
        com_vel = self.data.subtree_linvel[0]
        return com_pos, com_vel

    def compute_balance_torque(self):
        """Calculates 'Reflex' forces to keep the robot upright"""
        if not hasattr(self, 'com_ref'):
            self.com_ref = self.data.subtree_com[0].copy()
            self.com_ref[2] = 0.0 

        com_pos, com_vel = self.get_com_state()
        
        # Goal: Keep CoM over the feet (Anchor Point)
        target_pos = self.com_ref.copy()
        target_pos[2] = com_pos[2] 
        
        f_x = COM_KP[0] * (target_pos[0] - com_pos[0]) - COM_KD[0] * com_vel[0]
        f_y = COM_KP[1] * (target_pos[1] - com_pos[1]) - COM_KD[1] * com_vel[1]
        f_z = 0.0
        
        virtual_force = np.array([f_x, f_y, f_z])
        
        mujoco.mj_jacSubtreeCom(self.model, self.data, self.jac_com, 0)
        balance_torques_full = self.jac_com.T @ virtual_force
        
        return balance_torques_full[6:18] # Apply ONLY to legs

    def get_observations(self):
        base_quat = self.data.qpos[3:7]
        base_ang_vel = torch.tensor(self.data.qvel[3:6], dtype=torch.float32) * 0.25
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        
        # INVERT GRAVITY (World frame relative to Body)
        proj_grav = torch.tensor(r.inv().apply([0, 0, -1]), dtype=torch.float32)
        
        current_dof_pos = torch.tensor(self.data.qpos[7:7+29], dtype=torch.float32)
        current_dof_vel = torch.tensor(self.data.qvel[6:6+29], dtype=torch.float32)
        
        leg_pos = (current_dof_pos[LEG_INDICES] - self.default_dof_pos) * 1.0
        leg_vel = current_dof_vel[LEG_INDICES] * 0.05
        commands = torch.zeros(3) 
        sin_cos = torch.tensor([0.0, 1.0]) 
        
        obs = torch.cat([base_ang_vel, proj_grav, commands, leg_pos, leg_vel, self.last_actions, sin_cos])
        return obs.unsqueeze(0)

    def run(self, url):
        self.receiver.connect(url)
        print("⏳ Waiting for Mocap Data...")
        while not self.receiver.get_latest(): time.sleep(0.1)
        
        print(f"⚡ Starting PURE RL LEGS Loop")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Init Pose
            if self.model.nkey > 0:
                self.data.qpos[:] = self.model.key_qpos[0]
            else:
                self.data.qpos[:] = 0.0
                self.data.qpos[3] = 1.0
                self.data.qpos[7:19] = self.default_dof_pos.numpy()

            self.data.qpos[2] = 0.85 # Height
            mujoco.mj_forward(self.model, self.data)
            self.com_ref = self.data.subtree_com[0].copy()

            step = 0
            # Initialize Targets
            rl_leg_targets = self.default_dof_pos.numpy() 
            upper_body_targets = np.zeros(17) 

            while viewer.is_running():
                start = time.time()
                
                # --- 1. LEGS: UPDATED BY RL (50Hz) ---
                if step % DECIMATION == 0:
                    with torch.no_grad():
                        obs = self.get_observations()
                        actions = self.policy(obs).detach()[0]
                        self.last_actions = actions
                        rl_leg_targets = (actions.numpy() * 0.5) + self.default_dof_pos.numpy()

                # --- 2. UPPER BODY: UPDATED BY VR (MOCAP) ---
                mocap_data = self.receiver.get_latest()
                if mocap_data:
                    # Convert Mocap to Robot Joint Angles
                    full_qpos = self.converter.convert_to_qpos(mocap_data, self.data.qpos)
                    # STRICTLY extract only indices 12-29 (Waist + Arms)
                    upper_body_targets = full_qpos[7+12 : 7+29]

                # --- 3. PHYSICS CALCULATION ---
                # A. Leg Torques (RL + Balance Helper)
                qpos_legs = self.data.qpos[7 : 7+12]
                qvel_legs = self.data.qvel[6 : 6+12]
                
                balance_correction = self.compute_balance_torque()
                
                leg_torques = LEG_KP * (rl_leg_targets - qpos_legs) - LEG_KD * qvel_legs
                leg_torques += balance_correction # Add the helper force

                # B. Arm Torques (VR Tracking)
                qpos_arms = self.data.qpos[7+12 : 7+29]
                qvel_arms = self.data.qvel[6+12 : 6+29]
                
                arm_torques = ARM_KP * (upper_body_targets - qpos_arms) - ARM_KD * qvel_arms

                # --- 4. APPLY TO ROBOT ---
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="wss://improvable-maile-unquerulously.ngrok-free.dev",
        help="Server URL"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="policies\policy_lstm_1.pt",
        help="Path to policy file"
    )
    parser.add_argument("--xml", type=str, default="xml/scene.xml", help="Path to scene.xml")
    args = parser.parse_args()
    print(args.url)
    print(args.policy)
    sim = G1Controller(args.xml, args.policy)
    sim.run(args.url)