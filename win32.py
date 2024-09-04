from pathlib import Path
import struct
import io
import datetime
import pandas as pd
import numpy as np
# Win32 data class
# two options to import data
# 1. from the file path: Win32Data(file_path)
# 2. from the binary data: Win32Data.read_bin(bin_data)
# Both options will return the Win32Data object including self.win32_bin_data as binary data

class Win32Data():
    
    def __init__(self, file_path=None, bin_data=None, calib_coeff=1, num_channel=3):
        
        self.calib_coeff = calib_coeff
        self.num_channel = num_channel
        
        if file_path:
            self.header, self.data_int, self.data_float = self.set_file_path(file_path)
        
        elif bin_data:
            self.header, self.data_int, self.data_float = self.read_bin_data(bin_data)
        
    
    def set_file_path(self, file_path):
        self.file_path = Path(file_path).resolve()
        
        with open(self.file_path, "rb") as f:
            temp_bin = f.read()
        
        return self.read_bin_data(temp_bin)

    
    def read_bin_data(self, bin_data):
        
        bin_data = io.BytesIO(bin_data) 
        self.header = {}
        self.header["WIN32 Header"] = bin_data.read(4).hex()
        
        temp_data = []
                
        while True:
            # validate if the file is ended
            try:
                temp_start_datetime = struct.unpack(">8s", bin_data.read(8))[0].hex()
            except struct.error:
                break
            else:
                try:
                    temp_start_datetime_dt = datetime.datetime.strptime(temp_start_datetime, "%Y%m%d%H%M%S%f")
                except ValueError:
                    raise ValueError("Invalid datetime format")
                
            temp_second_block = {
                "start_datetime": temp_start_datetime,
                "frame_length": struct.unpack(">I", bin_data.read(4))[0],
                "channel_data_block_length": struct.unpack(">I", bin_data.read(4))[0],
            }
            
            temp_second_block_data = []
            
            for i in range(self.num_channel):
  
                temp_data_block = {
                    "org_id": struct.unpack(">b", bin_data.read(1))[0],
                    "net_id": struct.unpack(">b", bin_data.read(1))[0],
                    "channel_id": struct.unpack(">h", bin_data.read(2))[0],
                }
                
                temp_bin = bin_data.read(2)
                temp_data_block["sample_size"] = int(bin(temp_bin[0]>>4), 0)
                temp_data_block["data_size_per_data_block"] = struct.unpack(">H", temp_bin)[0] & 0x0FFF
                temp_data_block_sample = []
                
                temp_sample_value = struct.unpack(">i", bin_data.read(4))[0]
                temp_data_block_sample.append(temp_sample_value)
                
                for _ in range(temp_data_block["data_size_per_data_block"] - 1):
                    
                    if temp_data_block["sample_size"] == 0:
                        # 4bit float
                        raise ValueError("Not implemented for sample size 0")
                    elif temp_data_block["sample_size"] == 1:
                        temp_inc = struct.unpack(">b", bin_data.read(1))[0]
                    elif temp_data_block["sample_size"] == 2:
                        temp_inc = struct.unpack(">h", bin_data.read(2))[0]
                    elif temp_data_block["sample_size"] == 3:
                        temp_inc_bit = bin_data.read(3)
                        temp_inc = struct.unpack(">i", temp_inc_bit + b'\x00')[0]
                    elif temp_data_block["sample_size"] == 4:
                        temp_inc = struct.unpack(">i", bin_data.read(4))[0]
                        
                    temp_sample_value += temp_inc
                    temp_data_block_sample.append(temp_sample_value)
                
                temp_data_block["data"] = temp_data_block_sample
                temp_second_block_data.append(temp_data_block)
            
            temp_second_block["data"] = temp_second_block_data
            temp_data.append(temp_second_block)
        
        temp_data_int, temp_data_float = self._verify_data(temp_data)
        
        return (self.header, temp_data_int, temp_data_float)
        
    def _verify_data(self, data):
        
        self.header["start_datatime"] = data[0]["start_datetime"]
        self.header["end_datetime"] = data[-1]["start_datetime"]
        self.header["calib_coeff"] = self.calib_coeff
        self.header["num_channel"] = self.num_channel
        
        temp_data = [data[0]["data"][i]["data"]for i in range(self.num_channel)]
        temp_data = np.array(temp_data)
        
        for i in range(1,len(data)):
            
            temp_data_block = [data[i]["data"][j]["data"] for j in range(self.num_channel)]
            temp_data_block = np.array(temp_data_block)
            temp_data = np.hstack((temp_data, temp_data_block))
            
        temp_data_int = temp_data.T
        temp_data_float = temp_data_int * self.calib_coeff
        
        return (temp_data_int, temp_data_float)
    
    
    