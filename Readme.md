# Rebocap to Mujoco

This repo only uses 12 rebocap sensors:  
```
[ 'Chest', 'Waist', 'Right upper arm', 'Right lower arm', 'Right hand', 'Left Upper arm', 'Left lower arm', 'Left hand', 'Right thigh', 'Right calf', 'Left thigh', 'left calf' ]
```
## Folder Structure:
```
rebocap_g1_realtime/
├── config.py  
├── main.py   
├── main_2.py   
│            
├── realtime_mujoco_viewer.py    
│
├── webreceiver.py
├── webserver.py
│
├── scene.xml
├── g1_29dof.xml                 
├── meshes/                      
└── rebocap_ws_sdk/              
```

## Steps
### Run via direct SDK connection:
```
python realtime_mujoco_viewer.py --model_path scene.xml --calibrate
```  
### To make configuration easier, the above script is split into "main.py" and "config.py" 
- Configurations can be made in "config.py"  
- "main.py" includes locked hips and ground clamping for the feet, not included originally.
```
python main.py --model_path scene.xml --calibrate
```  
- "main_2.py includes unlocked hips.
```
python main_2.py --model_path scene.xml --calibrate
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
