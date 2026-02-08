#!/usr/bin/env python3
"""
WEBSERVER (Ngrok Version) - WITH TOKEN
"""
import asyncio
import websockets
import json
import threading
import queue
import time
import sys

# Import ReboCap SDK
try:
    import rebocap_ws_sdk
except ImportError:
    print("❌ ReboCap SDK not found. Make sure 'rebocap_ws_sdk' folder is here.")
    sys.exit(1)

# Import Ngrok
try:
    from pyngrok import ngrok, conf
except ImportError:
    print("❌ 'pyngrok' not found. Please run: pip install pyngrok")
    sys.exit(1)

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
# 1. Set your Auth Token here 
NGROK_AUTH_TOKEN = "36j4Xx7PhcMQ47Bq73htV2KNCQO_5oHMuwoPc2yGrKRbMrpct"

# 2. Ports
LOCAL_SDK_PORT = 7690
BROADCAST_PORT = 8080
# ==========================================

class ReboCapNgrokServer:
    def __init__(self):
        self.q = queue.Queue(maxsize=2)
        self.clients = set()
        
        # Configure Ngrok with your token immediately
        print(f"🔑 Setting Ngrok Authtoken...")
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        
        # Initialize ReboCap
        self.sdk = rebocap_ws_sdk.RebocapWsSdk(
            coordinate_type=rebocap_ws_sdk.CoordinateType.UnityCoordinate,
            use_global_rotation=True
        )
        self.sdk.set_pose_msg_callback(self._on_pose)

    def _on_pose(self, sdk, tran, pose24, static, ts):
        if not pose24: return
        msg = json.dumps({
            "tran": list(tran) if tran else [0]*3,
            "pose24": [list(q) for q in pose24],
            "timestamp": ts
        })
        if self.q.full():
            try: self.q.get_nowait()
            except: pass
        self.q.put(msg)

    async def ws_handler(self, websocket):
        self.clients.add(websocket)
        try:
            while True:
                msg = self.q.get()
                await websocket.send(msg)
                await asyncio.sleep(0.001)
        except: pass
        finally: self.clients.discard(websocket)

    def start_tunnel(self):
        print(f"\n🚀 Starting Ngrok Tunnel on port {BROADCAST_PORT}...")
        
        try:
            # Equivalent to command line: "ngrok http 8080"
            public_url = ngrok.connect(BROADCAST_PORT, "http").public_url
            
            # Convert URL to wss:// for the viewer
            ws_url = public_url.replace("http", "ws").replace("https", "wss")
            
            print(f"\n" + "="*60)
            print(f"✅ PUBLIC URL: \033[92m{ws_url}\033[0m")
            print("   (Copy this to webreceiver.py on your remote laptop)")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Ngrok Error: {e}")
            print("   (Check if your token is correct or if ngrok is already running)")

    def run(self):
        print(f"🔌 Connecting to Sensors on port {LOCAL_SDK_PORT}...")
        if self.sdk.open(LOCAL_SDK_PORT) != 0:
            print("❌ Failed to connect to ReboCap software.")
            return

        # Start Tunnel
        threading.Thread(target=self.start_tunnel, daemon=True).start()

        # Start Server
        print(f"📡 WebSocket Server running locally on {BROADCAST_PORT}...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(self.ws_handler, "0.0.0.0", BROADCAST_PORT)
        
        try:
            loop.run_until_complete(start_server)
            loop.run_forever()
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            ngrok.kill()

if __name__ == "__main__":
    ReboCapNgrokServer().run()