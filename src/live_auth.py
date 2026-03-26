import time
import sys
import os
import signal
import numpy as np
import threading
import queue

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import ECGAuthenticator

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Error: 'pyserial' library is required for live data.")
    print("Please run: pip install pyserial")
    sys.exit(1)

# --- Configuration ---
BAUD_RATE = 115200       # Adjust to match your Arduino/MAX30003 sketch
SAMPLE_RATE = 200        # Target sample rate (Hz)
BUFFER_SIZE = 1000       # 5 seconds @ 200Hz
Serial_Port = None       # Will be selected continuously

# Global control
running = True
data_queue = queue.Queue()

def signal_handler(sig, frame):
    global running
    print("\nStopping...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def find_arduino():
    ports = list(list_ports.comports())
    for p in ports:
        # Heuristic to find common serial devices
        if "USB" in p.description or "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return None

def serial_reader(port_name):
    """
    Runs in a separate thread. Expects binary ProtoCentral Packet (19 bytes).
    """
    global running
    
    print(f"Connecting to {port_name} @ {BAUD_RATE}...")
    try:
        with serial.Serial(port_name, BAUD_RATE, timeout=0.1) as ser:
            print("Connected. Listening for ProtoCentral packets...")
            
            # Flush existing buffer
            ser.reset_input_buffer()
            
            last_time = time.time()
            count = 0
            
            while running:
                if ser.in_waiting >= 19:
                    # Try to find header 0x0A, 0xFA
                    try:
                        while True:
                            if ser.in_waiting < 19:
                                time.sleep(0.005)
                                break

                            # Keep reading 1 byte until we find 0x0A
                            byte1 = ser.read(1)
                            if not byte1: break 
                            
                            if byte1 == b'\x0A':
                                byte2 = ser.read(1)
                                if byte2 == b'\xFA':
                                    # Found Header! Read rest of 19-byte packet (17 bytes left)
                                    packet_rest = ser.read(17)
                                    if len(packet_rest) != 17:
                                        break 
                                        
                                    # ProtoCentral structure:
                                    # Byte 0: 0x0A
                                    # Byte 1: 0xFA
                                    # Byte 2: 0x0C (Packet Length) -> packet_rest[0]
                                    # Byte 3: 0x00 (Packet ID)     -> packet_rest[1]
                                    # Byte 4: 0x02 (Data ID)       -> packet_rest[2]
                                    # Byte 5-8: ECG Data (LITTLE ENDIAN) -> packet_rest[3:7]
                                    
                                    ecg_bytes = packet_rest[3:7]
                                    val = int.from_bytes(ecg_bytes, byteorder='little', signed=True)
                                    
                                    data_queue.put(val)
                                    
                                    # Rate monitoring
                                    count += 1
                                    if count >= 125: # Pico sends approx 125Hz
                                        now = time.time()
                                        if (now - last_time) >= 30.0: # Print every 30s
                                            hz = count / (now - last_time)
                                            # print(f"  [DEBUG] Rate: {hz:.1f} Hz")
                                            count = 0
                                            last_time = now
                                    
                                    # Success, break to check buffer again
                                    break 
                                else:
                                    # Not 0xFA, maybe it was 0x0A data?
                                    # Continue scanning
                                    pass 
                    except Exception as e:
                        # print(f"Parse Error: {e}")
                        pass
                        
    except Exception as e:
        print(f"Serial Error: {e}")
        running = False

def enroll_live(auth, user_id=None, samples_needed=3):
    if not user_id:
        user_id = input("Enter User ID to enroll: ").strip() or "Owner"

    print(f"\n--- ENROLLMENT MODE: {user_id} ---")
    print("Please place your fingers on the sensor and relax.")
    print(f"Recording {samples_needed} segments (approx {samples_needed * 5} sec) to build profile...")
    
    collected_segments = []
    current_buffer = []
    
    while len(collected_segments) < samples_needed and running:
        if not data_queue.empty():
            val = data_queue.get()
            current_buffer.append(val)
            
            if len(current_buffer) >= BUFFER_SIZE:
                # We have a full segment
                seg = np.array(current_buffer)
                collected_segments.append(seg)
                current_buffer = [] # Clear buffer
                print(f"  [ENROLLING] Capture {len(collected_segments)} of {samples_needed} complete.")
    
    if collected_segments:
        print("Enrolling...")
        success = auth.enroll_user(user_id, collected_segments)
        if success:
            print("Enrollment Successful!")
            auth.save_profiles()
            return True
    return False

def authenticate_live(auth, user_id="Owner"):
    print(f"\n--- AUTHENTICATION MODE: Validating against '{user_id}' ---")
    print("Monitoring live ECG... (decisions every 5 seconds)")
    
    current_buffer = []
    
    while running:
        if not data_queue.empty():
            val = data_queue.get()
            current_buffer.append(val)
            
            # Sliding window or tumbling window?
            # Tumbling (non-overlapping) is simpler for demo
            if len(current_buffer) >= BUFFER_SIZE:
                seg = np.array(current_buffer)
                
                # Check authentication
                is_match, score = auth.authenticate(user_id, seg, threshold=0.70)
                
                status_icon = "🔓 ACCESS GRANTED" if is_match else "🔒 ACCESS DENIED"
                print(f"Score: {score:.4f}  |  {status_icon}")
                
                # Clear buffer (overlap=0)
                current_buffer = [] 

if __name__ == "__main__":
    # 1. Init Model
    if not os.path.exists("ecg_model.pth"):
        print("Error: Model not found. Please run src/train.py first.")
        sys.exit(1)
        
    auth = ECGAuthenticator()
    
    # 2. Setup Serial
    port = find_arduino()
    if not port:
        print("No Serial Port found automatically.")
        port = input("Enter COM port manually (e.g., COM3): ")
    
    # Start Serial Thread
    reader_thread = threading.Thread(target=serial_reader, args=(port,), daemon=True)
    reader_thread.start()
    
    print("\nWaiting for signal stability (3 seconds)...")
    time.sleep(3)
    
    # Load any existing profiles
    auth.load_profiles()
    
    # Auto-load from data directory if profiles are empty or user asks?
    # Let's just do it
    auth.load_from_data_folder("data/users")
    
    # 3. Mode Selection
    print("\nOptions:")
    print("1. Enroll (Record your 'Self' profile from live sensor)")
    print("2. Authenticate (Verify live user against enrolled profile)")
    choice = input("Select mode (1/2): ")
    
    if choice == "1":
        enroll_live(auth) # Will prompt for name now
        
        cont = input("Enrollment done. Switch to Authentication? (y/n): ")
        if cont.lower() == 'y':
            # Ask which user just enrolled
            users = list(auth.enrolled_embeddings.keys())
            target_user = users[-1] if users else "Owner"
            authenticate_live(auth, user_id=target_user)
            
    elif choice == "2":
        if not auth.enrolled_embeddings:
           print("No enrolled users found. Please enroll first.")
           enroll_live(auth)
           # Get the user we just enrolled
           users = list(auth.enrolled_embeddings.keys())
           user_to_auth = users[-1] if users else "Owner"
           authenticate_live(auth, user_id=user_to_auth)
        else:
           # Select user
           users = list(auth.enrolled_embeddings.keys())
           print("\nRegistered Users:")
           for i, u in enumerate(users):
               print(f"{i+1}. {u}")
           
           if len(users) == 1:
               selected_user = users[0]
               print(f"Authenticating against only user: {selected_user}")
           else:
               idx_str = input(f"Select user to verify against (1-{len(users)}): ")
               try:
                   idx = int(idx_str) - 1
                   if 0 <= idx < len(users):
                       selected_user = users[idx]
                   else:
                       print("Invalid selection. Defaulting to first user.")
                       selected_user = users[0]
               except:
                   print("Invalid input. Defaulting to first user.")
                   selected_user = users[0]
           
           authenticate_live(auth, user_id=selected_user)
