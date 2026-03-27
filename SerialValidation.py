import sys
import serial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import SignalProcessing  # Imports from your provided file

# ====================================================================================
# CONFIGURATION
# ====================================================================================
COM_PORT = 'COM9'
BAUD_RATE = 115200

# The expected sampling rate of your Arduino in Hz. 
FS = 1000.0  

# Configure the time window here (in milliseconds)
VIS_WINDOW_MS = 1000  
NUM_CHANNELS = 6

# Calculate the number of samples to keep in the buffer
BUFFER_SIZE = int((VIS_WINDOW_MS / 1000.0) * FS)

# ====================================================================================
# SERIAL READING THREAD
# ====================================================================================
class SerialReaderThread(QtCore.QThread):
    """Reads serial data in the background to prevent the UI from freezing."""
    new_data = QtCore.pyqtSignal(list)

    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return

        while self.running:
            try:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                
                # Parse the Arduino format: "Min:0,Max:1023,A0:val,A1:val..."
                parts = line.split(',')
                channel_vals = []
                for p in parts:
                    if ':' in p:
                        key, val = p.split(':')
                        if key.startswith('A') and key[1:].isdigit():
                            channel_vals.append(float(val))
                
                # Only emit if we captured all expected channels
                if len(channel_vals) == NUM_CHANNELS:
                    self.new_data.emit(channel_vals)
            except Exception as e:
                pass # Ignore occasional corrupted serial lines

    def stop(self):
        self.running = False
        self.wait()

# ====================================================================================
# MAIN GUI APPLICATION
# ====================================================================================
class RealTimePlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MyoWare Live DSP Plotter")
        self.resize(1200, 700)

        # Initialize data buffer (Channels x Samples)
        self.data_buffer = np.zeros((NUM_CHANNELS, BUFFER_SIZE))
        
        # Trackers for global Min and Max
        self.overall_min = float('inf')
        self.overall_max = float('-inf')

        self.init_ui()
        
        # Start the Serial Thread
        self.serial_thread = SerialReaderThread(COM_PORT, BAUD_RATE)
        self.serial_thread.new_data.connect(self.update_buffer)
        self.serial_thread.start()

        # UI Update Timer (Updates plot at ~30 FPS to save CPU)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)

    def init_ui(self):
        # Main Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # --- LEFT: Plot Area ---
        self.plot_widget = pg.PlotWidget(title=f"Live sEMG Feed ({VIS_WINDOW_MS} ms)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        
        # Add Horizontal Lines for Min and Max
        # --- Compatibility fix for PyQt5 vs PyQt6 Enums ---
        try:
            dash_style = QtCore.Qt.PenStyle.DashLine
        except AttributeError:
            dash_style = QtCore.Qt.DashLine
            
        # Add Horizontal Lines for Min and Max
        self.max_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', width=2, style=dash_style))
        self.min_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2, style=dash_style))
        self.plot_widget.addItem(self.max_line)
        self.plot_widget.addItem(self.min_line)
        self.plot_widget.addItem(self.max_line)
        self.plot_widget.addItem(self.min_line)

        main_layout.addWidget(self.plot_widget, stretch=4)

        # Plot Curves (Distinct colors for 6 channels)
        colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255), 
                  (255, 255, 50), (255, 50, 255), (50, 255, 255)]
        self.curves = []
        for i in range(NUM_CHANNELS):
            curve = self.plot_widget.plot(pen=pg.mkPen(color=colors[i], width=2), name=f"A{i}")
            self.curves.append(curve)

        # --- RIGHT: Control Panel ---
        control_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=1)
        
        # Plot Settings
        plot_settings_group = QtWidgets.QGroupBox("Plot Settings")
        plot_settings_layout = QtWidgets.QVBoxLayout()
        
        self.cb_minmax = QtWidgets.QCheckBox("Show Min/Max Bounds")
        self.cb_minmax.setChecked(True)
        plot_settings_layout.addWidget(self.cb_minmax)
        
        self.btn_reset_bounds = QtWidgets.QPushButton("Reset Min/Max History")
        self.btn_reset_bounds.clicked.connect(self.reset_bounds)
        plot_settings_layout.addWidget(self.btn_reset_bounds)
        
        plot_settings_group.setLayout(plot_settings_layout)
        control_panel.addWidget(plot_settings_group)

        # Channel Visibility toggles
        chan_group = QtWidgets.QGroupBox("Channel Visibility")
        chan_layout = QtWidgets.QVBoxLayout()
        self.chan_checks = []
        for i in range(NUM_CHANNELS):
            cb = QtWidgets.QCheckBox(f"Show Channel A{i}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.reset_bounds) # Reset bounds if a channel is hidden/shown
            self.chan_checks.append(cb)
            chan_layout.addWidget(cb)
        chan_group.setLayout(chan_layout)
        control_panel.addWidget(chan_group)

        # Filters Group
        filter_group = QtWidgets.QGroupBox("DSP Filters (Applied to all)")
        filter_layout = QtWidgets.QVBoxLayout()

        # Notch Filter
        self.cb_notch = QtWidgets.QCheckBox("Enable 50Hz Notch Filter")
        self.cb_notch.stateChanged.connect(self.reset_bounds)
        filter_layout.addWidget(self.cb_notch)
        
        # Bandpass Filter
        self.cb_bp = QtWidgets.QCheckBox("Enable Bandpass Filter")
        self.cb_bp.stateChanged.connect(self.reset_bounds)
        filter_layout.addWidget(self.cb_bp)
        
        # Bandpass Sliders/Spinboxes
        bp_config_layout = QtWidgets.QFormLayout()
        self.spin_low = QtWidgets.QSpinBox()
        self.spin_low.setRange(1, int(FS/2 - 2))
        self.spin_low.setValue(30)
        self.spin_low.valueChanged.connect(self.reset_bounds)
        
        self.spin_high = QtWidgets.QSpinBox()
        self.spin_high.setRange(2, int(FS/2 - 1)) # Max is Nyquist frequency
        self.spin_high.setValue(min(450, int(FS/2 - 1)))
        self.spin_high.valueChanged.connect(self.reset_bounds)
        
        bp_config_layout.addRow("Low Cut (Hz):", self.spin_low)
        bp_config_layout.addRow("High Cut (Hz):", self.spin_high)
        filter_layout.addLayout(bp_config_layout)

        # Rectification
        self.cb_rectify = QtWidgets.QCheckBox("Enable Rectification")
        self.cb_rectify.stateChanged.connect(self.reset_bounds)
        filter_layout.addWidget(self.cb_rectify)

        filter_group.setLayout(filter_layout)
        control_panel.addWidget(filter_group)
        control_panel.addStretch() # Push everything to the top
        
    def reset_bounds(self):
        """Resets the global min/max trackers. Called when filters are toggled."""
        self.overall_min = float('inf')
        self.overall_max = float('-inf')

    def update_buffer(self, new_vals):
        """Called whenever the serial thread reads a new full line."""
        self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
        for i in range(NUM_CHANNELS):
            self.data_buffer[i, -1] = new_vals[i]

    def update_plot(self):
        """Applies filters, updates curves, and auto-scales the Y-axis."""
        current_window_min = float('inf')
        current_window_max = float('-inf')
        
        # We need a small amount of data to avoid math errors on startup
        buffer_primed = not np.all(self.data_buffer[:, 0] == 0)

        for i in range(NUM_CHANNELS):
            if not self.chan_checks[i].isChecked():
                self.curves[i].setData([])
                continue
            
            signal = self.data_buffer[i, :].copy()

            if buffer_primed:
                try:
                    if self.cb_notch.isChecked():
                        signal = SignalProcessing.notchFilter(signal, fs=FS, notchFreq=50.0)

                    if self.cb_bp.isChecked():
                        low = self.spin_low.value()
                        high = self.spin_high.value()
                        if low < high: 
                            signal = SignalProcessing.bandpassFilter(signal, fs=FS, lowCut=low, highCut=high)

                    if self.cb_rectify.isChecked():
                        signal = SignalProcessing.rectifySignal(signal)
                        
                except Exception as e:
                    pass # Fallback to raw if filter has edge-effect crash

            # Track min/max of the current data being shown
            ch_min = np.min(signal)
            ch_max = np.max(signal)
            if ch_min < current_window_min: current_window_min = ch_min
            if ch_max > current_window_max: current_window_max = ch_max

            self.curves[i].setData(signal)

        # Update the visual bounds and scale the Y-Axis
        if current_window_min != float('inf') and buffer_primed:
            # Expand the global history
            self.overall_min = min(self.overall_min, current_window_min)
            self.overall_max = max(self.overall_max, current_window_max)
            
            if self.cb_minmax.isChecked():
                # Show lines and scale the plot to hold the history bounds steady
                self.min_line.setValue(self.overall_min)
                self.max_line.setValue(self.overall_max)
                self.min_line.show()
                self.max_line.show()
                
                padding = (self.overall_max - self.overall_min) * 0.05
                if padding == 0: padding = 1.0 # Prevent division by zero errors
                self.plot_widget.setYRange(self.overall_min - padding, self.overall_max + padding)
            else:
                # Hide lines and scale the plot tightly to the *current* 1000ms window
                self.min_line.hide()
                self.max_line.hide()
                
                cur_padding = (current_window_max - current_window_min) * 0.05
                if cur_padding == 0: cur_padding = 1.0
                self.plot_widget.setYRange(current_window_min - cur_padding, current_window_max + cur_padding)

    def closeEvent(self, event):
        self.serial_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimePlotter()
    window.show()
    sys.exit(app.exec())