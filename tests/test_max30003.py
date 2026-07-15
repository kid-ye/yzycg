import serial
from serial.tools import list_ports
import time
import struct

def find_serial_port():
    """Find available serial ports"""
    ports = list(list_ports.comports())
    print("Available Serial Ports:")
    for i, p in enumerate(ports):
        print(f"{i+1}. {p.device} - {p.description}")
    return ports

def test_max30003(port_name, baud_rate=115200):
    """Test MAX30003 sensor connectivity and data"""
    print(f"\n=== MAX30003 Diagnostic Test ===")
    print(f"Connecting to {port_name} @ {baud_rate}...")
    
    try:
        with serial.Serial(port_name, baud_rate, timeout=1) as ser:
            print("✓ Serial connection established")
            
            # Flush buffer
            ser.reset_input_buffer()
            time.sleep(1)
            
            print("\nWaiting for data packets...")
            packets_received = 0
            valid_packets = 0
            start_time = time.time()
            
            while time.time() - start_time < 10:  # Test for 10 seconds
                if ser.in_waiting >= 19:
                    # Look for header
                    byte1 = ser.read(1)
                    if byte1 == b'\x0A':
                        byte2 = ser.read(1)
                        if byte2 == b'\xFA':
                            # Found valid header!
                            packet_rest = ser.read(17)
                            if len(packet_rest) == 17:
                                packets_received += 1
                                
                                # Extract ECG value
                                ecg_bytes = packet_rest[3:7]
                                ecg_val = int.from_bytes(ecg_bytes, byteorder='little', signed=True)
                                
                                # Extract RR interval
                                rr_bytes = packet_rest[7:9]
                                rr_ms = int.from_bytes(rr_bytes, byteorder='little', signed=False)
                                
                                # Extract Heart Rate
                                hr_bytes = packet_rest[11:13]
                                hr_bpm = int.from_bytes(hr_bytes, byteorder='little', signed=False)
                                
                                # Check if data looks reasonable
                                if -200000 < ecg_val < 200000:
                                    valid_packets += 1
                                
                                # Print every 25 packets (~0.2 seconds)
                                if packets_received % 25 == 0:
                                    print(f"Packet #{packets_received}: ECG={ecg_val:7d}  HR={hr_bpm:3d} bpm  RR={rr_ms:4d} ms")
            
            # Results
            elapsed = time.time() - start_time
            print(f"\n=== Test Results ===")
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Packets Received: {packets_received}")
            print(f"Valid Packets: {valid_packets}")
            print(f"Data Rate: {packets_received/elapsed:.1f} Hz")
            
            if packets_received == 0:
                print("\n❌ FAILED: No data received")
                print("   - Check if pico_helper.py is running on Pico")
                print("   - Check USB connection")
                print("   - Verify correct COM port")
            elif valid_packets < packets_received * 0.8:
                print("\n⚠️  WARNING: Many invalid packets")
                print("   - Check sensor connections (SPI wiring)")
                print("   - Check MAX30003 power supply")
            else:
                print("\n✓ SUCCESS: MAX30003 is working properly!")
                print("   - Sensor is streaming data")
                print("   - Ready for live authentication")
            
            return packets_received > 0
            
    except serial.SerialException as e:
        print(f"❌ Serial Error: {e}")
        print("   - Check if port is already in use")
        print("   - Try unplugging and replugging the Pico")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("MAX30003 Sensor Diagnostic Tool\n")
    
    # List available ports
    ports = find_serial_port()
    
    if not ports:
        print("\n❌ No serial ports found!")
        print("   - Connect your Raspberry Pi Pico")
        print("   - Install USB drivers if needed")
        exit(1)
    
    # Select port
    if len(ports) == 1:
        selected_port = ports[0].device
        print(f"\nAuto-selecting: {selected_port}")
    else:
        choice = input(f"\nSelect port (1-{len(ports)}): ")
        try:
            idx = int(choice) - 1
            selected_port = ports[idx].device
        except:
            print("Invalid selection")
            exit(1)
    
    # Run test
    test_max30003(selected_port)
