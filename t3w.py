from pathlib import Path
import struct
from .win32 import Win32Data
import json
import pandas as pd

class t3wData():
    def __init__(self, file_path, calib_coeff=2.048 / 2 ** 23):
        
        self.file_path = Path(file_path).resolve()
        self.calib_coeff = calib_coeff
        
        # Read the t3w file
        self._read_t3w_file()
    
    def export_header(self, dir_path=None):
        t3w_header = self.t3w_header
        
        if not dir_path:
            dir_path = self.file_path.parent
        
        temp_file_path = Path(dir_path).resolve() / (self.file_path.stem + "_header.json")
        
        with open(temp_file_path, "w") as f:
            json.dump(t3w_header, f, indent=4)
        
        return temp_file_path
    
    def export_data_csv(self, dir_path=None, time_format="relative"):
        
        if not dir_path:
            dir_path = self.file_path.parent
        
        temp_data = pd.DataFrame(self.t3w_win32_data.get_data_float())
        temp_data.columns = [0, 1, 2]
        temp_data["relative_time"] = temp_data.index * self.t3w_header["sampling_time_interval"] / 1000
        temp_data["absolute_time"] = pd.to_datetime(self.t3w_header["start_datetime_this_file"], format="%Y%m%d%H%M%S%f") \
            + pd.to_timedelta(temp_data["relative_time"], unit="s")
            
        # change order of columns
        if time_format == "relative":
            temp_data = temp_data[["relative_time", 0, 1, 2]]
        elif time_format == "absolute":
            temp_data = temp_data[["absolute_time", 0, 1, 2]]
        elif time_format == "both":
            temp_data = temp_data[["absolute_time", "relative_time", 0, 1, 2]]
        
        temp_file_path = Path(dir_path).resolve() / (self.file_path.stem + "_data.csv")
        temp_data.to_csv(temp_file_path, index=False)
        
        return temp_file_path
        
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

        temp_t3w_win32_data = Win32Data(bin_data=t3w_bin_data_win32, calib_coeff=self.calib_coeff)
        temp_t3w_win32_data_header = temp_t3w_win32_data.get_header()
        
        print(temp_t3w_win32_data_header)
        
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
        
        temp_t3w_header["latitude"] *= -1 if temp_t3w_header["north_south_flag"] == "S" else 1
        temp_t3w_header["longitude"] *= -1 if temp_t3w_header["east_west_flag"] == "W" else 1
        
        return temp_t3w_header
        