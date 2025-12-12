#!/usr/bin/env python3
"""
Configuration for ReboCap MuJoCo G1 Integration
"""

# ============================================
# MODEL AND CONNECTION SETTINGS
# ============================================
MODEL_PATH = "g1_29dof.xml"
REBOCAP_PORT = 7690
ROOT_HEIGHT_OFFSET = 1.0  # Height offset in meters 

# ============================================
# CALIBRATION SETTINGS
# ============================================
CALIBRATION_WAIT_TIME = 10  # Seconds to wait before capturing. Only works for main_2.py

# ============================================
# COORDINATE SYSTEM CONFIGURATION
# ============================================
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