import tkinter as tk
import socket

# --- Network Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
UPDATE_FREQUENCY_HZ = 50  # 50 Hz = 20ms delay between packets
DELAY_MS = int(1000 / UPDATE_FREQUENCY_HZ)

# Initialize UDP Socket (Fire-and-forget protocol)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Helper Function for Text Input ---
def update_slider(event, slider, entrybox, min_val, max_val):
    try:
        # Get the value typed by the user
        val = float(entrybox.get())
        
        # Clamp the value to the min/max bounds
        clamped_val = max(min_val, min(val, max_val))
        
        # Update the slider (which automatically updates the telemetry loop)
        slider.set(clamped_val)
    except ValueError:
        # If they type letters or garbage, ignore the update
        pass
    finally:
        # Always clear the text box after Enter is pressed
        entrybox.delete(0, tk.END)

# --- Helper Function for Preset Buttons ---
def set_all_sliders(yaw, pitch, roll, elbow):
    """Instantly sets all sliders to specific values"""
    yaw_slider.set(yaw)
    pitch_slider.set(pitch)
    roll_slider.set(roll)
    elbow_slider.set(elbow)

# --- GUI Setup ---
root = tk.Tk()
root.title("Prosthetic Twin - Manual Override")
# Increased window size to fit the new button grid
root.geometry("550x650") 
root.configure(padx=20, pady=20)

# --- Control Sections ---
# Section 1: Yaw
yaw_frame = tk.Frame(root)
yaw_frame.pack(fill='x', pady=5)
yaw_slider = tk.Scale(yaw_frame, from_=-180, to=180, orient='horizontal', label='Shoulder Flexion / Extension (Yaw Forward/Back)') 
yaw_slider.pack(side='left', fill='x', expand=True)
yaw_entry = tk.Entry(yaw_frame, width=7)
yaw_entry.pack(side='right', padx=5, pady=25)
yaw_entry.bind('<Return>', lambda event: update_slider(event, yaw_slider, yaw_entry, -180, 180))

# Section 2: Pitch
pitch_frame = tk.Frame(root)
pitch_frame.pack(fill='x', pady=5)
pitch_slider = tk.Scale(pitch_frame, from_=-180, to=180, orient='horizontal', label='Shoulder Abduction / Adduction (Pitch Up/Down)')
pitch_slider.pack(side='left', fill='x', expand=True)
pitch_entry = tk.Entry(pitch_frame, width=7)
pitch_entry.pack(side='right', padx=5, pady=25)
pitch_entry.bind('<Return>', lambda event: update_slider(event, pitch_slider, pitch_entry, -180, 180))

# Section 3: Roll
roll_frame = tk.Frame(root)
roll_frame.pack(fill='x', pady=5)
roll_slider = tk.Scale(roll_frame, from_=-180, to=180, orient='horizontal', label='Shoulder Internal / External Rotation (Roll Twist)')
roll_slider.pack(side='left', fill='x', expand=True)
roll_entry = tk.Entry(roll_frame, width=7)
roll_entry.pack(side='right', padx=5, pady=25)
roll_entry.bind('<Return>', lambda event: update_slider(event, roll_slider, roll_entry, -180, 180))

# Section 4: Elbow
elbow_frame = tk.Frame(root)
elbow_frame.pack(fill='x', pady=5)
elbow_slider = tk.Scale(elbow_frame, from_=0, to=150, orient='horizontal', label='Elbow Flexion')
elbow_slider.pack(side='left', fill='x', expand=True)
elbow_entry = tk.Entry(elbow_frame, width=7)
elbow_entry.pack(side='right', padx=5, pady=25)
elbow_entry.bind('<Return>', lambda event: update_slider(event, elbow_slider, elbow_entry, 0, 150))

# --- Preset Movements Section ---
preset_frame = tk.Frame(root)
preset_frame.pack(fill='x', pady=20)

# Configure the 4 columns to expand and space evenly
for i in range(4):
    preset_frame.columnconfigure(i, weight=1)

# Row 0: Movements 1 to 4
tk.Button(preset_frame, text="Movement 1", command=lambda: set_all_sliders(45, 0, 0, 0)).grid(row=0, column=0, padx=5, pady=5, sticky='we')
tk.Button(preset_frame, text="Movement 2", command=lambda: set_all_sliders(90, 0, 0, 0)).grid(row=0, column=1, padx=5, pady=5, sticky='we')
tk.Button(preset_frame, text="Movement 3", command=lambda: set_all_sliders(110, 0, 0, 0)).grid(row=0, column=2, padx=5, pady=5, sticky='we')
tk.Button(preset_frame, text="Movement 4", command=lambda: set_all_sliders(-30, 0, 0, 0)).grid(row=0, column=3, padx=5, pady=5, sticky='we')

# Row 1: Movements 5 to 8
tk.Button(preset_frame, text="Movement 5", command=lambda: set_all_sliders(0, 45, 0, 0)).grid(row=1, column=0, padx=5, pady=5, sticky='we')
tk.Button(preset_frame, text="Movement 6", command=lambda: set_all_sliders(0, 90, 0, 0)).grid(row=1, column=1, padx=5, pady=5, sticky='we')
tk.Button(preset_frame, text="Movement 7", command=lambda: set_all_sliders(0, 0, 40, 0)).grid(row=1, column=2, padx=5, pady=5, sticky='we')
tk.Button(preset_frame, text="Movement 8", command=lambda: set_all_sliders(0, 0, 90, 0)).grid(row=1, column=3, padx=5, pady=5, sticky='we')

# Row 2: Zero All Angles (Spans across all 4 columns)
tk.Button(preset_frame, text="Zero All Angles", command=lambda: set_all_sliders(0, 0, 0, 0)).grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky='we')

# --- Telemetry Loop ---
def send_telemetry():
    # 1. Grab current values from sliders
    yaw = float(yaw_slider.get())
    pitch = float(pitch_slider.get())
    roll = float(roll_slider.get())
    elbow = float(elbow_slider.get())
    
    # 2. Format exactly as the Unity script expects: "yaw,pitch,roll,elbow"
    packet_string = f"{yaw},{pitch},{roll},{elbow}"
    
    # 3. Send over UDP
    try:
        sock.sendto(packet_string.encode('utf-8'), (UDP_IP, UDP_PORT))
        print(f"[UDP] SENDING DATA - Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}, Elbow: {elbow}")
    except Exception as e:
        print(f"Network error: {e}")
        
    # 4. Schedule this function to run again automatically without blocking the GUI
    root.after(DELAY_MS, send_telemetry)

# Kick off the continuous sending loop
send_telemetry()

# Start the Tkinter window
root.mainloop()