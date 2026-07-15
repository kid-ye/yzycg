import machine
import time
import sys

# --- SPI Setup ---
spi = machine.SPI(0, baudrate=1000000, polarity=0, phase=0,
                  sck=machine.Pin(18), mosi=machine.Pin(19), miso=machine.Pin(16))

cs = machine.Pin(17, machine.Pin.OUT)
cs.value(1)

# --- MAX30003 Core Register Addresses ---
SW_RST     = 0x08
SYNCH      = 0x09
FIFO_RST   = 0x0A
CNFG_GEN   = 0x10
CNFG_CAL   = 0x12
CNFG_EMUX  = 0x14
CNFG_ECG   = 0x15
CNFG_RTOR1 = 0x1D
ECG_FIFO   = 0x21
RTOR       = 0x25
STATUS     = 0x01  # Status register for FIFO checks

def write_register(address, data):
    try:
        cs.value(0)
        addr_byte = (address << 1) & 0xFE
        spi.write(bytearray([addr_byte, (data >> 16) & 0xFF, (data >> 8) & 0xFF, data & 0xFF]))
        cs.value(1)
        return True
    except Exception as e:
        cs.value(1)
        return False

def read_register(address):
    try:
        cs.value(0)
        addr_byte = (address << 1) | 0x01
        spi.write(bytearray([addr_byte]))
        data = spi.read(3)
        cs.value(1)
        return (data[0] << 16) | (data[1] << 8) | data[2]
    except Exception as e:
        cs.value(1)
        return None

# --- Initialization Sequence ---
write_register(SW_RST, 0x000000)
time.sleep(0.1)

# Configure with better settings for accuracy
write_register(CNFG_GEN, 0x080000)   # Enable ECG channel
write_register(CNFG_CAL, 0x000000)   # Disable calibration
write_register(CNFG_EMUX, 0x000000)  # Normal ECG input
write_register(CNFG_ECG, 0x805000)   # Gain=20V/V, Sample Rate=128sps
write_register(CNFG_RTOR1, 0x3FC600) # Enable R-to-R detection
write_register(FIFO_RST, 0x000000)   # Reset FIFO
write_register(SYNCH, 0x000000)      # Start conversion

# --- Packet Structure (19 Bytes) ---
packet = bytearray([
    0x0A, 0xFA, 0x0C, 0x00, 0x02, # [0-4]   Header
    0, 0, 0, 0,                   # [5-8]   ECG Data
    0, 0,                         # [9-10]  RR Interval
    0, 0,                         # [11-12] Padding
    0, 0,                         # [13-14] Heart Rate
    0, 0,                         # [15-16] Padding
    0x00, 0x0B                    # [17-18] Footer
])

# State variables
hr_bpm = 0
rr_ms = 0
error_count = 0
MAX_ERRORS = 10

print("Streaming to OpenView. Do not print anything else!")

while True:
    # 1. Read ECG FIFO with validation
    raw_data = read_register(ECG_FIFO)
    
    if raw_data is None:
        error_count += 1
        if error_count > MAX_ERRORS:
            # Reset device
            write_register(SW_RST, 0x000000)
            time.sleep(0.1)
            write_register(SYNCH, 0x000000)
            error_count = 0
        continue
    
    error_count = 0
    
    # 2. Extract and sign-extend 18-bit ECG value (bits 23:6)
    ecg_val = (raw_data >> 6) & 0x3FFFF  # Mask to 18 bits
    
    # Sigīn extension for 18-bit two's complement
    if ecg_val & 0x20000:  # Check sign bit (bit 17)
        ecg_val -= 0x40000  # Convert to negative
    
    # 3. Pack ECG data (32-bit little endian)
    packet[5] = ecg_val & 0xFF
    packet[6] = (ecg_val >> 8) & 0xFF
    packet[7] = (ecg_val >> 16) & 0xFF
    packet[8] = (ecg_val >> 24) & 0xFF
    
    # 4. Read R-to-R data
    rtor_data = read_register(RTOR)
    
    if rtor_data is not None:
        # Extract 14-bit interval (bits 23:10)
        rtor_raw = (rtor_data >> 10) & 0x3FFF
        
        # Update HR/RR only on valid beat detection
        if rtor_raw > 0:
            # Calculate heart rate: 60 / (interval_in_seconds)
            # interval_in_seconds = rtor_raw / 128 (128 Hz internal clock)
            # HR = 60 * 128 / rtor_raw = 7680 / rtor_raw
            hr_bpm = int(7680 / rtor_raw)
            
            # Calculate RR interval in milliseconds
            rr_ms = int((rtor_raw * 1000) / 128)
            
            # Sanity check: HR should be between 30-220 bpm
            if hr_bpm < 30 or hr_bpm > 220:
                hr_bpm = 0
                rr_ms = 0
    
    # 5. Pack RR interval (16-bit little endian)
    packet[9] = rr_ms & 0xFF
    packet[10] = (rr_ms >> 8) & 0xFF
    
    # 6. Pack heart rate (16-bit little endian)
    packet[13] = hr_bpm & 0xFF
    packet[14] = (hr_bpm >> 8) & 0xFF
    
    # 7. Send packet
    sys.stdout.buffer.write(packet)
    
    # 8. Sampling rate control
    # For 128 sps (matching CNFG_ECG setting): 1000ms / 128 ≈ 7.8ms
    # For 200 sps (matching preprocessing): 1000ms / 200 = 5ms
    # For 1000 sps (matching PTBDB): 1000ms / 1000 = 1ms
    time.sleep(0.008)  # 125 Hz - adjust based on your needs
