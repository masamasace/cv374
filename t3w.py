from pathlib import Path
import struct
from .win32 import Win32Data

class t3wData():
    def __init__(self, file_path):
        
        self.file_path = Path(file_path).resolve()
        
        # Read the t3w file
        self._read_t3w_file()
        
    def _read_t3w_file(self):
        
        with open(self.file_path, "rb") as f:
            temp_t3w_bin_data = f.read()
            
            temp_t3w_bin_data_header = temp_t3w_bin_data[:1024]
            temp_t3w_bin_data_win32 = temp_t3w_bin_data[1024:]
            
            # Read the header
            self.t3w_header = self._read_t3w_header(temp_t3w_bin_data_header)
            
            # Read the win32 data
            self.t3w_win32_data = self._read_win32_data(temp_t3w_bin_data_win32, self.t3w_header)

    
    def _read_win32_data(self, t3w_bin_data_win32, t3w_header):

        temp_t3w_win32_data = Win32Data(bin_data=t3w_bin_data_win32, calib_coeff=2.048 / 2 ** 23)
        
        return temp_t3w_win32_data
        
    def _read_t3w_header(self, t3w_bin_data_header):
        
        temp_t3w_header = {
            "device_program_name": struct.unpack(">12s", t3w_bin_data_header[4:16])[0].decode("utf-8"),
            "device_number": struct.unpack(">H", t3w_bin_data_header[24:26])[0],
            "num_channel": struct.unpack(">H", t3w_bin_data_header[30:32])[0],
            "sampling_time_interval": struct.unpack(">H", t3w_bin_data_header[40:42])[0],
            "delay_time": struct.unpack(">H", t3w_bin_data_header[42:44])[0],
            "sequence_number": struct.unpack(">H", t3w_bin_data_header[50:52])[0],
            "start_datetime_this_file": str(struct.unpack(">H", t3w_bin_data_header[52:54])[0]).zfill(2) + str(struct.unpack(">H", t3w_bin_data_header[54:56])[0]).zfill(2) \
                                        + str(struct.unpack(">H", t3w_bin_data_header[56:58])[0]).zfill(2) + str(struct.unpack(">H", t3w_bin_data_header[58:60])[0]).zfill(2) \
                                        + str(struct.unpack(">H", t3w_bin_data_header[60:62])[0]).zfill(2) + str(struct.unpack(">H", t3w_bin_data_header[62:64])[0]).zfill(2) \
                                        + str(struct.unpack(">H", t3w_bin_data_header[64:66])[0]).zfill(2),
            "start_datetime_first_file": str(struct.unpack(">H", t3w_bin_data_header[66:68])[0]).zfill(2) + str(struct.unpack(">H", t3w_bin_data_header[68:70])[0]).zfill(2) \
                                        + str(struct.unpack(">H", t3w_bin_data_header[70:72])[0]).zfill(2) + str(struct.unpack(">H", t3w_bin_data_header[72:74])[0]).zfill(2) \
                                        + str(struct.unpack(">H", t3w_bin_data_header[74:76])[0]).zfill(2) + str(struct.unpack(">H", t3w_bin_data_header[76:78])[0]).zfill(2) \
                                        + str(struct.unpack(">H", t3w_bin_data_header[78:80])[0]).zfill(2),
            "channel_offset_1": struct.unpack(">I", t3w_bin_data_header[224:228])[0],
            "channel_offset_2": struct.unpack(">I", t3w_bin_data_header[228:232])[0],
            "channel_offset_3": struct.unpack(">I", t3w_bin_data_header[232:236])[0],
            "latitude" : struct.unpack(">I", t3w_bin_data_header[808:812])[0] + struct.unpack(">I", t3w_bin_data_header[812:816])[0]/60,
            "longitude" : struct.unpack(">I", t3w_bin_data_header[816:820])[0] + struct.unpack(">I", t3w_bin_data_header[820:824])[0]/60,
            "north_south_flag": struct.unpack(">c", t3w_bin_data_header[828:829])[0].decode("utf-8"),
            "east_west_flag": struct.unpack(">c", t3w_bin_data_header[829:830])[0].decode("utf-8")
        }
        
        return temp_t3w_header
        