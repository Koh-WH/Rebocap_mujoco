#!/usr/bin/env python3
"""
Configuration for ReboCap MuJoCo G1 Integration - GLOBAL ROTATION MODE
Specifies parent-child sensor relationships for independent joint control
"""

# ============================================
# ROOT ROTATION SETTINGS
# ============================================
# Control which axes from pelvis sensor affect root rotation
ROOT_ROTATION_CONFIG = {
    'use_yaw': True,    # Turning left/right
    'use_pitch': False, # Leaning forward/back (can cause slanting if True)
    'use_roll': False   # Leaning left/right (can cause slanting if True)
}

# ============================================
# MODEL AND CONNECTION SETTINGS
# ============================================
MODEL_PATH = "xml/scene.xml"
REBOCAP_PORT = 7690
ROOT_HEIGHT_OFFSET = 0.0  # Height offset in meters

# ============================================
# CALIBRATION SETTINGS
# ============================================
CALIBRATION_WAIT_TIME = 3  # Seconds to wait before capturing

# ============================================
# SMPL SENSOR INDICES (Reference)
# ============================================
"""
0:  Pelvis       9:  Spine3 (Chest)    16: L_Shoulder    
1:  L_Hip        10: L_Foot (unused)   17: R_Shoulder
2:  R_Hip        11: R_Foot (unused)   18: L_Elbow
3:  Spine1       12: Neck              19: R_Elbow
4:  L_Knee       13: L_Collar          20: L_Wrist
5:  R_Knee       14: R_Collar          21: R_Wrist
6:  Spine2       15: Head (unused)     22: L_Hand
7:  L_Ankle                            23: R_Hand
8:  R_Ankle
"""

# ============================================
# JOINT MAPPING WITH PARENT-CHILD RELATIONSHIPS
# ============================================
# Format:
# {
#     'joint': 'mujoco_joint_name',
#     'sensor': child_sensor_index,
#     'parent_sensor': parent_sensor_index (None for root-attached),
#     'axis': 'x' | 'y' | 'z',
#     'scale': float,
#     'offset': float (radians)
# }

REBOCAP_TO_G1_MAPPING = [
    # ========================================
    # TORSO/WAIST
    # ========================================
    # Waist controlled by Chest (Spine3) relative to Pelvis
    {'joint': 'waist_pitch_joint', 'sensor': 9, 'parent_sensor': 0, 'axis': 'x', 'scale': 0.8, 'offset': 0.0},
    {'joint': 'waist_yaw_joint', 'sensor': 0, 'parent_sensor': 0, 'axis': 'z', 'scale': 0.8, 'offset': 0.0},
    {'joint': 'waist_roll_joint', 'sensor': 0, 'parent_sensor': 0, 'axis': 'y', 'scale': -0.6, 'offset': 0.0},
    
    # ========================================
    # LEFT LEG
    # ========================================
    # Hip: L_Hip sensor (1) relative to Pelvis (0)
    {'joint': 'left_hip_pitch_joint', 'sensor': 1, 'parent_sensor': 0, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    {'joint': 'left_hip_roll_joint', 'sensor': 1, 'parent_sensor': 0, 'axis': 'z', 'scale': 1.0, 'offset': 0.2},
    {'joint': 'left_hip_yaw_joint', 'sensor': 1, 'parent_sensor': 0, 'axis': 'y', 'scale': -1.0, 'offset': 0.0},
    
    # Knee: L_Knee sensor (4) relative to L_Hip (1)
    {'joint': 'left_knee_joint', 'sensor': 4, 'parent_sensor': 1, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    
    # Ankle: L_Ankle sensor (7) relative to L_Knee (4)
    {'joint': 'left_ankle_pitch_joint', 'sensor': 7, 'parent_sensor': 4, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    {'joint': 'left_ankle_roll_joint', 'sensor': 7, 'parent_sensor': 4, 'axis': 'z', 'scale': 0.5, 'offset': 0.0},
    
    # ========================================
    # RIGHT LEG
    # ========================================
    # Hip: R_Hip sensor (2) relative to Pelvis (0)
    {'joint': 'right_hip_pitch_joint', 'sensor': 2, 'parent_sensor': 0, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    {'joint': 'right_hip_roll_joint', 'sensor': 2, 'parent_sensor': 0, 'axis': 'z', 'scale': 1.0, 'offset': -0.2},
    {'joint': 'right_hip_yaw_joint', 'sensor': 2, 'parent_sensor': 0, 'axis': 'y', 'scale': -1.0, 'offset': 0.0},
    
    # Knee: R_Knee sensor (5) relative to R_Hip (2)
    {'joint': 'right_knee_joint', 'sensor': 5, 'parent_sensor': 2, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    
    # Ankle: R_Ankle sensor (8) relative to R_Knee (5)
    {'joint': 'right_ankle_pitch_joint', 'sensor': 8, 'parent_sensor': 5, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    {'joint': 'right_ankle_roll_joint', 'sensor': 8, 'parent_sensor': 5, 'axis': 'z', 'scale': 0.5, 'offset': 0.0},
    
    # ========================================
    # LEFT ARM
    # ========================================
    # Shoulder: L_Shoulder sensor (16) relative to Chest (9)
    {'joint': 'left_shoulder_pitch_joint', 'sensor': 16, 'parent_sensor': 9, 'axis': 'y', 'scale': -1.0, 'offset': 0.0},
    {'joint': 'left_shoulder_roll_joint', 'sensor': 16, 'parent_sensor': 9, 'axis': 'z', 'scale': -1.0, 'offset': 1.2},
    {'joint': 'left_shoulder_yaw_joint', 'sensor': 16, 'parent_sensor': 9, 'axis': 'x', 'scale': -1.0, 'offset': 0.0},
    
    # Elbow: L_Elbow sensor (18) relative to L_Shoulder (16)
    {'joint': 'left_elbow_joint', 'sensor': 18, 'parent_sensor': 16, 'axis': 'x', 'scale': 0.8, 'offset': 1.2},
    
    # Wrist: L_Wrist sensor (20) relative to L_Elbow (18)
    {'joint': 'left_wrist_pitch_joint', 'sensor': 20, 'parent_sensor': 18, 'axis': 'y', 'scale': -0.8, 'offset': 0.0},
    {'joint': 'left_wrist_roll_joint', 'sensor': 20, 'parent_sensor': 18, 'axis': 'x', 'scale': 0.8, 'offset': 0.0},
    {'joint': 'left_wrist_yaw_joint', 'sensor': 20, 'parent_sensor': 18, 'axis': 'z', 'scale': -0.8, 'offset': 0.0},
    
    # ========================================
    # RIGHT ARM
    # ========================================
    # Shoulder: R_Shoulder sensor (17) relative to Chest (9)
    {'joint': 'right_shoulder_pitch_joint', 'sensor': 17, 'parent_sensor': 9, 'axis': 'y', 'scale': 1.0, 'offset': 0.0},
    {'joint': 'right_shoulder_roll_joint', 'sensor': 17, 'parent_sensor': 9, 'axis': 'z', 'scale': -1.0, 'offset': -1.2},
    {'joint': 'right_shoulder_yaw_joint', 'sensor': 17, 'parent_sensor': 9, 'axis': 'x', 'scale': 1.0, 'offset': 0.0},
    
    # Elbow: R_Elbow sensor (19) relative to R_Shoulder (17)
    {'joint': 'right_elbow_joint', 'sensor': 19, 'parent_sensor': 17, 'axis': 'x', 'scale': 0.8, 'offset': 1.2},
    
    # Wrist: R_Wrist sensor (21) relative to R_Elbow (19)
    {'joint': 'right_wrist_pitch_joint', 'sensor': 21, 'parent_sensor': 19, 'axis': 'y', 'scale': 0.8, 'offset': 0.0},
    {'joint': 'right_wrist_roll_joint', 'sensor': 21, 'parent_sensor': 19, 'axis': 'x', 'scale': -0.8, 'offset': 0.0},
    {'joint': 'right_wrist_yaw_joint', 'sensor': 21, 'parent_sensor': 19, 'axis': 'z', 'scale': -0.8, 'offset': 0.0},
]

# ============================================
# NOTES
# ============================================
"""
PARENT-CHILD RELATIONSHIPS:

This config uses GLOBAL rotations from sensors, then computes
relative rotations between parent and child sensors.

Example: Elbow joint
- Child sensor: 18 (L_Elbow) in global space
- Parent sensor: 16 (L_Shoulder) in global space
- Joint rotation = L_Elbow rotation RELATIVE TO L_Shoulder rotation

This gives INDEPENDENT control - moving shoulder doesn't affect
elbow joint value, only the elbow sensor itself controls elbow joint.

HIERARCHY:
Pelvis (0)
├─ L_Hip (1)
│  └─ L_Knee (4)
│     └─ L_Ankle (7)
├─ R_Hip (2)
│  └─ R_Knee (5)
│     └─ R_Ankle (8)
└─ Chest/Spine3 (9)
   ├─ L_Shoulder (16)
   │  └─ L_Elbow (18)
   │     └─ L_Wrist (20)
   └─ R_Shoulder (17)
      └─ R_Elbow (19)
         └─ R_Wrist (21)

TUNING:
- 'axis': Which rotation to extract ('x'=roll, 'y'=pitch, 'z'=yaw)
- 'scale': Sensitivity (negative to reverse direction)
- 'offset': Neutral position adjustment in radians
- Try different axes if movement is on wrong axis
- Adjust scale if too weak/strong or reversed
"""