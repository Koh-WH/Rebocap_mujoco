#!/usr/bin/env python3
"""
TELEOP DANCE: UNIVERSAL RL LEGS CONTROLLER
- Supports ANY Policy Size (12-DoF, 29-DoF, etc.)
- LEGS (0-11): Controlled by RL Policy
- UPPER BODY (12+): Controlled by VR/Mocap
- AUTO-DETECTS policy input size on startup
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
# The G1 robot has 29 joints total.
# We explicitly define which indices correspond to what.
LEG_INDICES = list(range(12))            # Joints 0-11 (Legs)
UPPER_BODY_INDICES = list(range(12, 29)) # Joints 12-28 (Waist + Arms + Head)

# 1. GAINS
LEG_KP = 100.0    
LEG_KD = 10.0     
ARM_KP = 40.0
ARM_KD = 1.0

# 2. BALANCE ARBITRATION
COM_KP = [150.0, 150.0, 0.0] 
COM_KD = [10.0, 10.0, 0.0]

# 3. TIMING
DECIMATION = 10  

class G1Controller:
    def __init__(self, model_path, policy_path):
        print(f"📂 Loading XML: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"🧠 Loading RL Policy: {policy_path}")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # --- AUTO-DETECT POLICY DOF ---
        self.policy_dof = self._detect_policy_dof()
        print(f"🔍 Detected Policy Structure: {self.policy_dof}-DoF")
        
        self.receiver = RemoteReceiver()
        self.converter = MotionConverter(self.model)
        
        # Action buffer matches the policy size
        self.last_actions = torch.zeros(self.policy_dof)
        
        # --- INITIALIZE FULL BODY DEFAULT POSE (29 JOINTS) ---
        # We need a full 29-element vector even if the policy only uses 12.
        self.default_dof_pos = torch.zeros(29)
        
        if self.model.nkey > 0:
            # Load from XML keyframe if available
            self.default_dof_pos = torch.from_numpy(self.model.key_qpos[0][7:7+29]).float()
        else:
            # Fallback: Bent knees for legs, zeros for arms
            bent_knees = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, # L Leg
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0  # R Leg
            ], dtype=np.float32)
            self.default_dof_pos[0:12] = torch.from_numpy(bent_knees)
            
        self.jac_com = np.zeros((3, self.model.nv))

    def _detect_policy_dof(self):
        """
        Tries to run the policy with different input sizes to determine N.
        Standard IsaacGym Obs: 
        BaseVel(3) + Gravity(3) + Cmd(3) + DofPos(N) + DofVel(N) + LastAct(N) + Clock(2)
        Input Size = 11 + 3*N
        """
        possible_dofs = [12, 23, 29] # Common G1 configurations
        
        for dof in possible_dofs:
            input_dim = 11 + (3 * dof)
            try:
                # Try a dummy forward pass
                dummy_obs = torch.zeros(1, input_dim)
                self.policy(dummy_obs)
                return dof
            except Exception:
                continue
        
        print("⚠️  WARNING: Could not auto-detect DoF. Defaulting to 12.")
        return 12

    def get_com_state(self):
        com_pos = self.data.subtree_com[0]
        com_vel = self.data.subtree_linvel[0]
        return com_pos, com_vel

    def compute_balance_torque(self):
        if not hasattr(self, 'com_ref'):
            self.com_ref = self.data.subtree_com[0].copy()
            self.com_ref[2] = 0.0 

        com_pos, com_vel = self.get_com_state()
        target_pos = self.com_ref.copy()
        target_pos[2] = com_pos[2] 
        
        f_x = COM_KP[0] * (target_pos[0] - com_pos[0]) - COM_KD[0] * com_vel[0]
        f_y = COM_KP[1] * (target_pos[1] - com_pos[1]) - COM_KD[1] * com_vel[1]
        virtual_force = np.array([f_x, f_y, 0.0])
        
        mujoco.mj_jacSubtreeCom(self.model, self.data, self.jac_com, 0)
        balance_torques_full = self.jac_com.T @ virtual_force
        return balance_torques_full[6:18] 

    def get_observations(self):
        # 1. Base State
        base_quat = self.data.qpos[3:7]
        base_ang_vel = torch.tensor(self.data.qvel[3:6], dtype=torch.float32) * 0.25
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        proj_grav = torch.tensor(r.inv().apply([0, 0, -1]), dtype=torch.float32)
        
        # 2. Joint State (Dynamically sliced based on Policy DoF)
        # We assume the policy always wants the FIRST 'N' joints of the robot.
        N = self.policy_dof
        
        current_dof_pos = torch.tensor(self.data.qpos[7:7+N], dtype=torch.float32)
        current_dof_vel = torch.tensor(self.data.qvel[6:6+N], dtype=torch.float32)
        
        # Normalize using the default pose (sliced to N)
        target_dof_pos = (current_dof_pos - self.default_dof_pos[:N]) * 1.0
        target_dof_vel = current_dof_vel * 0.05
        
        commands = torch.zeros(3) 
        sin_cos = torch.tensor([0.0, 1.0]) 
        
        # 3. Construct Obs
        obs = torch.cat([
            base_ang_vel, 
            proj_grav, 
            commands, 
            target_dof_pos, 
            target_dof_vel, 
            self.last_actions, 
            sin_cos
        ])
        return obs.unsqueeze(0)

    def run(self, url):
        self.receiver.connect(url)
        print("⏳ Waiting for Mocap Data...")
        while not self.receiver.get_latest(): time.sleep(0.1)
        
        print(f"⚡ Starting Controller (Hybrid: RL Legs + Mocap Arms)")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Reset
            if self.model.nkey > 0:
                self.data.qpos[:] = self.model.key_qpos[0]
            else:
                self.data.qpos[:] = 0.0
                self.data.qpos[3] = 1.0
                # Initialize full body to defaults
                self.data.qpos[7:7+29] = self.default_dof_pos.numpy()

            self.data.qpos[2] = 0.82
            mujoco.mj_forward(self.model, self.data)
            self.com_ref = self.data.subtree_com[0].copy()

            step = 0
            
            # Target Arrays
            rl_leg_targets = self.default_dof_pos[0:12].numpy()
            upper_body_targets = self.default_dof_pos[12:29].numpy()

            while viewer.is_running():
                start = time.time()
                
                # --- 1. RL STEP (50Hz) ---
                if step % DECIMATION == 0:
                    with torch.no_grad():
                        obs = self.get_observations()
                        
                        # Policy Output (Size N)
                        actions = self.policy(obs).detach()[0]
                        self.last_actions = actions
                        
                        # We ONLY care about the first 12 actions (Legs)
                        # Even if policy outputs 29, we take [0:12] for legs
                        leg_actions = actions[0:12]
                        
                        rl_leg_targets = (leg_actions.numpy() * 0.5) + self.default_dof_pos[0:12].numpy()

                # --- 2. MOCAP UPDATE ---
                mocap_data = self.receiver.get_latest()
                if mocap_data:
                    full_qpos = self.converter.convert_to_qpos(mocap_data, self.data.qpos)
                    # Extract Upper Body (Indices 12-28)
                    upper_body_targets = full_qpos[7+12 : 7+29]

                # --- 3. PHYSICS & CONTROL ---
                
                # A. LEGS (RL + Balance Assist)
                qpos_legs = self.data.qpos[7 : 7+12]
                qvel_legs = self.data.qvel[6 : 6+12]
                balance_correction = self.compute_balance_torque()
                
                leg_torques = LEG_KP * (rl_leg_targets - qpos_legs) - LEG_KD * qvel_legs
                leg_torques += balance_correction

                # B. ARMS (Mocap PD)
                qpos_arms = self.data.qpos[7+12 : 7+29]
                qvel_arms = self.data.qvel[6+12 : 6+29]
                
                arm_torques = ARM_KP * (upper_body_targets - qpos_arms) - ARM_KD * qvel_arms

                # C. Apply to Full Robot
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
    parser.add_argument("--policy", type=str, required=True, help="Path to ANY .pt policy file (12 or 29 DoF)")
    parser.add_argument("--xml", type=str, default="xml/scene.xml")
    args = parser.parse_args()
    
    sim = G1Controller(args.xml, args.policy)
    sim.run(args.url)