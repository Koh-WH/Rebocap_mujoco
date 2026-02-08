# Rebocap to Mujoco  
This repo only uses 12 rebocap sensors:  
```
[ 'Chest', 'Waist', 'Right upper arm', 'Right lower arm', 'Right hand', 'Left Upper arm', 'Left lower arm', 'Left hand', 'Right thigh', 'Right calf', 'Left thigh', 'left calf' ]
```
## Folder Structure:  
```
rebocap_g1_realtime/
├── config_global.py  
├── main.py     
│            
├── realtime_mujoco_viewer.py    
│
├── webreceiver.py
├── webserver.py
├── RL_legs_webreceiver.py
│                
├── xml/                      
├── rebocap_ws_sdk/
├── policies/  
├── recordings/
│
├── test_play.py
└── test_record.py
```
## Table of contents 
1. [Direct SDK Connection](#Run-via-direct-SDK-connection)
2. [Ngrok](#Run-with-Ngrok)
3. [RL Policy](#With-RL-Policy-from-IsaacGym)
4. [Record & Play](#Recording-and-play)
5. [Useful Links](#Links)
  
### Run via direct SDK connection:
```
python realtime_mujoco_viewer.py --model_path scene.xml --calibrate
```  
### To make configuration easier, the above script is split into "main.py" and "config_global.py" 
- Configurations can be made in "config_global.py"  
- "main.py" includes unlocked hips and ground clamping for the feet.
```
python main.py --model_path scene.xml --calibrate
```  
```
python main_2.py --help
```
  
### Run with Ngrok:
- Copy the token from 'https://dashboard.ngrok.com/get-started/setup/windows'  
- Paste in "webserver.py"  
- Run ```pip install pyngrok``` in terminal  
- Run ```python webserver.py``` and copy the url  
- Open another terminal and run ```python webreceiver.py --calibrate``` and paste the url.
- Or run ```python webreceiver.py --calibrate --url <paste url>```
  
### With RL Policy from [IsaacGym](https://github.com/Koh-WH/g1_isaacgym)
- With RL policy for legs in physics environment  
```
python RL_legs_webreceiver.py
```
  
### Recording and play:
- test_play.py
- test_record.py
  
## Links
Rebocap Link:  
- https://doc.rebocap.com/en_US/SDK/  
  
Unitree Link:  
- https://support.unitree.com/home/en/G1_developer  
  
Other githubs:  
- https://github.com/unitreerobotics/unitree_mujoco/tree/main  
- https://github.com/robocasa/robocasa/tree/main  
- https://github.com/google-deepmind/mujoco_playground/tree/main  
- https://github.com/anderspitman/awesome-tunneling  
- https://github.com/YanjieZe/awesome-humanoid-robot-learning  
  