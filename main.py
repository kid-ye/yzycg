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
CNFG_RTOR1 = 0x1D   # Hardware R-to-R Enable Register
ECG_FIFO   = 0x21
RTOR       = 0x25   # R-to-R Data Register

def write_register(address, data):
    cs.value(0)
    addr_byte = (address << 1) & 0xFE
    spi.write(bytearray([addr_byte, (data >> 16) & 0xFF, (data >> 8) & 0xFF, data & 0xFF]))
    cs.value(1)

def read_register(address):
    cs.value(0)
    addr_byte = (address << 1) | 0x01
    spi.write(bytearray([addr_byte]))
    data = spi.read(3)
    cs.value(1)
    return (data[0] << 16) | (data[1] << 8) | data[2]

# --- Initialization Sequence ---
write_register(SW_RST, 0x000000)
time.sleep(0.1)
write_register(CNFG_GEN, 0x080000)
write_register(CNFG_CAL, 0x000000)
write_register(CNFG_EMUX, 0x000000)
write_register(CNFG_ECG, 0x805000)
write_register(CNFG_RTOR1, 0x3FC600) # Wake up the internal Heart Rate detector
write_register(FIFO_RST, 0x000000)
write_register(SYNCH, 0x000000)

# --- EXACT ProtoCentral Packet Structure (19 Bytes) ---
packet = bytearray([
    0x0A, 0xFA, 0x0C, 0x00, 0x02, # [0-4]   Header
    0, 0, 0, 0,                   # [5-8]   ECG Data
    0, 0,                         # [9-10]  RR Interval
    0, 0,                         # [11-12] Padding
    0, 0,                         # [13-14] Heart Rate
    0, 0,                         # [15-16] Padding
    0x00, 0x0B                    # [17-18] Footer
])

# --- EXACT ProtoCentral Packet Structure (19 Bytes) ---
packet = bytearray([
    0x0A, 0xFA, 0x0C, 0x00, 0x02, # [0-4]   Header
    0, 0, 0, 0,                   # [5-8]   ECG Data
    0, 0,                         # [9-10]  RR Interval
    0, 0,                         # [11-12] Padding
    0, 0,                         # [13-14] Heart Rate
    0, 0,                         # [15-16] Padding
    0x00, 0x0B                    # [17-18] Footer
])

# Initialize these OUTSIDE the loop so they remember their values
hr_bpm = 0
rr_ms = 0

print("Streaming to OpenView. Do not print anything else!")

while True:
    # 1. Grab and Pack the Raw ECG Wave
    raw_data = read_register(ECG_FIFO)
    ecg_val = raw_data >> 6
    if ecg_val > 131071:      
        ecg_val -= 262144      
       
    packet[5] = ecg_val & 0xFF
    packet[6] = (ecg_val >> 8) & 0xFF
    packet[7] = (ecg_val >> 16) & 0xFF
    packet[8] = (ecg_val >> 24) & 0xFF
   
    # 2. Grab and Calculate the R-to-R Data
    rtor_data = read_register(RTOR)
    # The interval is stored in the top 14 bits of the 24-bit register
    rtor_raw = (rtor_data >> 10) & 0x3FFF
   
    # ONLY update if a valid beat is detected.
    # Notice we REMOVED the "else" block!
    if rtor_raw > 0:
        hr_bpm = int(7680 / rtor_raw)
        rr_ms = int((rtor_raw * 1000) / 128)
       
    # 3. Pack the RR and HR integers into the packet (Little Endian)
    packet[9] = rr_ms & 0xFF
    packet[10] = (rr_ms >> 8) & 0xFF
   
    packet[13] = hr_bpm & 0xFF
    packet[14] = (hr_bpm >> 8) & 0xFF
   
    # 4. Blast the data
    sys.stdout.buffer.write(packet)
    time.sleep(0.008)
