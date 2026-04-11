import sys
import time
import os
import spidev
import numpy as np
import scipy.io
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from abc import ABC, abstractmethod

import ControllerConfiguration as Config

# ====================================================================================
# ========================= HARDWARE READING MODE INTERFACE ==========================
# ====================================================================================

class SignalReadingMode(ABC):
    """
    Abstract base class for hardware signal reading modes.
    Defines the interface that both Continuous and Data Collection modes implement.
    """
    
    def __init__(self):
        self.spi = None
        self._initialized = False
        
    def _initialize_spi(self):
        """Initialize SPI connection to MCP3008 ADC."""
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        # 1.35 MHz is the standard safe speed for MCP3008 on Raspberry Pi
        self.spi.max_speed_hz = 1350000
        self._initialized = True
        print(f"SPI Bus Opened. Reading {Config.NUM_CHANNELS} channels at {Config.FS} Hz (8x multiplexed = {Config.FS * Config.NUM_CHANNELS} Hz effective).\n")
        
    def _read_adc_channel(self, channel):
        """
        Requests and formats data from a specific MCP3008 channel.
        
        Args:
            channel (int): ADC channel 0-7
            
        Returns:
            int: 10-bit value (0-1023)
        """
        if channel < 0 or channel > 7:
            return -1
        
        # 3-byte SPI transaction: [control_byte, address_byte, dummy_byte]
        adc_response = self.spi.xfer2([1, (8 + channel) << 4, 0])
        
        # Reconstruct the 10-bit result from response bytes
        data = ((adc_response[1] & 3) << 8) + adc_response[2]
        return data
        
    @abstractmethod
    def read_sample_chunk(self):
        """
        Read a chunk of samples from all channels.
        
        Returns:
            np.ndarray: Shape (num_channels, increment) array of ADC values
        """
        pass
        
    def cleanup(self):
        """Close SPI connection and cleanup resources."""
        if self.spi:
            self.spi.close()
            print("SPI Bus Closed Safely.")

# ====================================================================================
# ====================== CONTINUOUS READING MODE (NO STORAGE) =========================
# ====================================================================================

class ContinuousReadingMode(SignalReadingMode):
    """
    Real-time streaming mode without persistent memory storage.
    Used for live control and monitoring. Data flows through the controller
    but is not accumulated for later saving.
    """
    
    def __init__(self):
        super().__init__()
        self._initialize_spi()
        
    def read_sample_chunk(self):
        """
        Read a chunk of samples at 8kHz effective rate (1kHz sampling × 8 channels).
        Data is read and returned immediately without storage.
        
        Returns:
            np.ndarray: Shape (8, Config.INCREMENT) array of 10-bit ADC values
        """
        chunk = np.zeros((Config.NUM_CHANNELS, Config.INCREMENT), dtype=np.float32)
        
        # Sample delay for 8kHz effective rate across 8 channels
        # 1 / (1000 Hz × 8 channels) = 125 microseconds per sample
        sample_delay = 1.0 / (Config.FS * Config.NUM_CHANNELS)
        
        # Multiplex: read all 8 channels for each sample increment
        for s in range(Config.INCREMENT):
            for ch in range(Config.NUM_CHANNELS):
                chunk[ch, s] = self._read_adc_channel(ch)
            time.sleep(sample_delay)
            
        return chunk

# ====================================================================================
# ====================== DATA COLLECTION MODE (WITH STORAGE) ==========================
# ====================================================================================

class DataCollectionMode(SignalReadingMode):
    """
    Streaming mode with persistent data storage.
    Used for recording trials or datasets. Data is accumulated in memory
    and can be saved to .mat files in the same format as secondary_data.
    """
    
    def __init__(self, collection_name=None):
        super().__init__()
        self.collection_name = collection_name or "hardware_collection"
        self.collected_data = []
        self.start_time = None
        self.sample_count = 0
        self._initialize_spi()
        
    def read_sample_chunk(self):
        """
        Read a chunk and accumulate in memory for later saving.
        
        Returns:
            np.ndarray: Shape (8, Config.INCREMENT) array of 10-bit ADC values
        """
        chunk = np.zeros((Config.NUM_CHANNELS, Config.INCREMENT), dtype=np.float32)
        
        sample_delay = 1.0 / (Config.FS * Config.NUM_CHANNELS)
        
        # Multiplex: read all 8 channels for each sample increment
        for s in range(Config.INCREMENT):
            for ch in range(Config.NUM_CHANNELS):
                chunk[ch, s] = self._read_adc_channel(ch)
            time.sleep(sample_delay)
            
        # Store a raw copy before any downstream controller processing.
        self.collected_data.append(chunk.copy())
        self.sample_count += chunk.shape[1]
        
        return chunk
        
    def save_collection(self, output_path=None):
        """
        Save collected data to a .mat file, matching secondary_data format.
        
        Args:
            output_path (str): Path to save the .mat file. Defaults to
                              './hardware_collections/{collection_name}.mat'
        
        Returns:
            str: Path to the saved .mat file, or None if save failed
        """
        if not self.collected_data:
            print("Warning: No data collected to save.")
            return None
            
        if output_path is None:
            os.makedirs('./hardware_collections', exist_ok=True)
            output_path = f"./hardware_collections/{self.collection_name}.mat"
        
        # Concatenate all chunks horizontally (along time axis)
        full_data = np.concatenate(self.collected_data, axis=1)
        
        # Save in same format as secondary_data: EMGDATA key
        scipy.io.savemat(output_path, {'EMGDATA': full_data})
        
        print(f"[Data Collection] Saved {self.sample_count} samples ({self.sample_count/Config.FS:.2f}s) to {output_path}")
        return output_path
        
    def clear_collection(self):
        """Clear collected data from memory."""
        self.collected_data = []
        self.sample_count = 0
        print("[Data Collection] Buffer cleared.")
        
    def get_collection_stats(self):
        """
        Get information about the current collection.
        
        Returns:
            dict: Statistics including sample count, duration, and chunks
        """
        return {
            'sample_count': self.sample_count,
            'duration_seconds': self.sample_count / Config.FS,
            'num_chunks': len(self.collected_data),
            'memory_mb': (self.sample_count * Config.NUM_CHANNELS * 4) / (1024 * 1024)
        }

# ====================================================================================
# =============== VISUALIZATION HELPER (Optional, Separate Concern) ===================
# ====================================================================================

class HardwareSignalVisualizer:
    """
    Optional PyQtGraph visualization for hardware signal monitoring.
    Decoupled from the reading modes so that hardware can operate
    independently of visualization.
    """
    
    def __init__(self, reading_mode):
        self.reading_mode = reading_mode
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        
        self.win = pg.GraphicsLayoutWidget(show=True, title="MCP3008 Hardware Feed")
        self.win.resize(1000, 800)
        self.win.setWindowTitle('Hardware Signal Monitor')
        
        self.vis_window_size = Config.WINDOW_SIZE * 3  
        self.data_buffer = np.zeros((Config.NUM_CHANNELS, self.vis_window_size))
        
        self.plots = []
        self.curves = []
        
        # Setup plots for each channel
        for i in range(Config.NUM_CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            
            p.setLabel('left', Config.CHANNEL_MAP.get(i, f"Ch {i}"))
            
            # MCP3008 is a 10-bit ADC, meaning raw data ranges from 0 to 1023
            p.setYRange(0, 1024, padding=0)
            p.disableAutoRange(axis=pg.ViewBox.YAxis)

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
            
        # Print table header
        header = " | ".join([f"CH{i:1}" for i in range(Config.NUM_CHANNELS)])
        print(f" {header} ")
        print("-" * (7 * Config.NUM_CHANNELS))
        
    def update_visualization(self, chunk):
        """Update the display with new data chunk."""
        # Shift main array and append chunk
        self.data_buffer = np.roll(self.data_buffer, -Config.INCREMENT, axis=1)
        self.data_buffer[:, -Config.INCREMENT:] = chunk
        
        # Print latest sample row
        latest_samples = chunk[:, -1]
        row_string = " | ".join([f"{int(val):4}" for val in latest_samples])
        print(f" {row_string} ")
        
        # Update all curves
        for i in range(Config.NUM_CHANNELS):
            self.curves[i].setData(self.data_buffer[i])
            
    def run(self):
        """Start the visualization event loop."""
        print("Starting SPI Telemetry Visualization. Close the GUI window to exit.")
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._read_and_display)
        self.timer.start(Config.INCREMENT)
        
        try:
            sys.exit(self.app.exec())
        finally:
            self.reading_mode.cleanup()
            
    def _read_and_display(self):
        """Timer callback to read and visualize."""
        chunk = self.reading_mode.read_sample_chunk()
        self.update_visualization(chunk)

# ====================================================================================
# ======================== LEGACY COMPATIBILITY CLASS ==========================
# ====================================================================================

# ====================================================================================
# ======================== LEGACY COMPATIBILITY CLASS ==========================
# ====================================================================================

class HardwareSignalReader:
    """
    Legacy compatibility wrapper for existing code.
    Wraps ContinuousReadingMode with integrated visualization.
    New code should use ContinuousReadingMode or DataCollectionMode directly.
    """
    
    def __init__(self):
        self.reading_mode = ContinuousReadingMode()
        self.visualizer = HardwareSignalVisualizer(self.reading_mode)
        
    def sample_and_update(self):
        """Legacy method for Qt timer callback."""
        chunk = self.reading_mode.read_sample_chunk()
        self.visualizer.update_visualization(chunk)

    def run(self):
        """Start the Qt Event Loop and hardware timer."""
        self.visualizer.run()
        
if __name__ == "__main__":
    # === EXAMPLE 1: Visualization Mode (Legacy) ===
    # reader = HardwareSignalReader()
    # reader.run()
    
    # === EXAMPLE 2: Continuous Reading Mode ===
    print("Starting Continuous Reading Mode (no storage)...")
    continuous_mode = ContinuousReadingMode()
    
    for i in range(5):
        chunk = continuous_mode.read_sample_chunk()
        print(f"Batch {i+1}: Read {chunk.shape[1]} samples from {chunk.shape[0]} channels")
        print(f"  Shape: {chunk.shape}, Min: {chunk.min():.1f}, Max: {chunk.max():.1f}")
    
    continuous_mode.cleanup()
    
    # === EXAMPLE 3: Data Collection Mode ===
    # print("\nStarting Data Collection Mode...")
    # collection_mode = DataCollectionMode(collection_name="test_movement_5")
    # 
    # for i in range(50):
    #     chunk = collection_mode.read_sample_chunk()
    #     if (i + 1) % 10 == 0:
    #         stats = collection_mode.get_collection_stats()
    #         print(f"Iteration {i+1}: {stats['sample_count']} samples collected ({stats['duration_seconds']:.2f}s)")
    # 
    # collection_mode.save_collection()
    # collection_mode.cleanup()
