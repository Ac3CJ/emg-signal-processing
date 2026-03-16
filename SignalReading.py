import sys
import time
import spidev
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import ControllerConfiguration as Config

class HardwareSignalReader:
    def __init__(self):
        # 1. Initialize SPI Connection
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        # 1.35 MHz is the standard safe speed for MCP3008 on Raspberry Pi
        self.spi.max_speed_hz = 1350000 
        print(f"SPI Bus Opened. Reading {Config.NUM_CHANNELS} channels at {Config.FS} Hz.\n")

        # 2. Initialize Data Arrays
        self.vis_window_size = Config.WINDOW_SIZE * 3  
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, self.vis_window_size))
        
        # 3. Setup Terminal Table Header
        header = " | ".join([f"CH{i:1}" for i in range(Config.NUM_CHANNELS)])
        print(f" {header} ")
        print("-" * (7 * Config.NUM_CHANNELS)) # Scales dashes based on channel count

        # 4. Setup the GUI
        self.setup_gui()

    def setup_gui(self):
        """Initializes the PyQtGraph multi-channel visualizer."""
        self.app = QtWidgets.QApplication(sys.argv)
        
        self.win = pg.GraphicsLayoutWidget(show=True, title="MCP3008 Live SPI Feed")
        self.win.resize(1000, 800)
        self.win.setWindowTitle('Hardware Verification Feed')
        
        self.plots = []
        self.curves = []
        
        for i in range(Config.NUM_CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            
            p.setLabel('left', Config.CHANNEL_MAP.get(i, f"Ch {i}"))
            
            # MCP3008 is a 10-bit ADC, meaning raw data ranges strictly from 0 to 1023
            p.setYRange(0, 1024, padding=0)
            p.disableAutoRange(axis=pg.ViewBox.YAxis)

            # Hide X-axis for all but the bottom plot to keep it clean
            if i < Config.NUM_CHANNELS - 1:
                p.hideAxis('bottom')
            
            vLine = pg.InfiniteLine(angle=90, movable=False, pos=self.vis_window_size - Config.WINDOW_SIZE)
            try:
                dash_style = QtCore.Qt.PenStyle.DashLine
            except AttributeError:
                dash_style = QtCore.Qt.DashLine
            vLine.setPen(pg.mkPen(color='r', style=dash_style, width=2))
            p.addItem(vLine)
            
            curve = p.plot(pen=pg.mkPen(color=(50, 255, 150), width=1))
            self.plots.append(p)
            self.curves.append(curve)

    def read_adc_channel(self, channel):
        """Requests and formats data from a specific MCP3008 channel."""
        if channel < 0 or channel > 7:
            return -1
        
        # 3-byte SPI transaction
        adc_response = self.spi.xfer2([1, (8 + channel) << 4, 0])
        
        # Reconstruct the 10-bit result
        data = ((adc_response[1] & 3) << 8) + adc_response[2]
        return data

    def sample_and_update(self):
        """Triggered by QTimer. Reads a chunk of samples."""
        chunk = np.zeros((Config.NUM_CHANNELS, Config.INCREMENT))
        
        # Software timing delay
        sample_delay = 1.0 / (Config.FS * Config.NUM_CHANNELS)
        
        # Multiplex
        for s in range(Config.INCREMENT):
            for ch in range(Config.NUM_CHANNELS):
                chunk[ch, s] = self.read_adc_channel(ch)
            time.sleep(sample_delay)
            
        # Shift main array and append chunk
        self.data_buffer = np.roll(self.data_buffer, -Config.INCREMENT, axis=1)
        self.data_buffer[:, -Config.INCREMENT:] = chunk
        
        # --- NEW: Print the latest instantaneous sample to the terminal table ---
        latest_samples = chunk[:, -1]
        row_string = " | ".join([f"{int(val):4}" for val in latest_samples])
        print(f" {row_string} ")
        
        # Push array to GUI
        for i in range(Config.NUM_CHANNELS):
            self.curves[i].setData(self.data_buffer[i])

    def run(self):
        """Starts the Qt Event Loop and the hardware timer."""
        print("Starting SPI Telemetry. Close the GUI window to exit.")
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.sample_and_update)
        self.timer.start(Config.INCREMENT) 
        
        try:
            sys.exit(self.app.exec())
        finally:
            self.spi.close()
            print("SPI Bus Closed Safely.")

if __name__ == "__main__":
    reader = HardwareSignalReader()
    reader.run()