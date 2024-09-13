from pathlib import Path
import struct
import io
import datetime
import numpy as np
from obspy import Trace, Stream, UTCDateTime
# Win32 data class
# two options to import data
# 1. from the file path: Win32Handler(file_path)
# 2. from the binary data: Win32Handler.read_bin(bin_data)
# Both options will return the Win32Handler object including self.win32_bin_data as binary data


class Win32Handler():
    
    def __init__(self, file_path=None, bin_data=None, calib_coeff=1, num_channel=3, 
                 flag_debug=False, debug_params=None):
        
        self.calib_coeff = calib_coeff
        self.num_channel = num_channel
        self.flag_debug = flag_debug
        self.debug_params = debug_params
        if self.debug_params is not None:
            self.debug_channel = self.debug_params[0]
            self.debug_start_index = self.debug_params[1]
            self.debug_end_index = self.debug_params[2]
        else:
            self.debug_channel = None
            self.debug_start_index = None
            self.debug_end_index = None
        
        if file_path:
            self.header, self.stream = self.set_file_path(file_path)
        
        elif bin_data:
            self.header, self.stream = self.read_bin_data(bin_data)
    
    def get_header(self):
        return self.header
    
    def get_stream(self):
        return self.stream
        
    
    def set_file_path(self, file_path):
        self.file_path = Path(file_path).resolve()
        
        with open(self.file_path, "rb") as f:
            temp_bin = f.read()
        
        return self.read_bin_data(temp_bin)

    
    def read_bin_data(self, bin_data):
        
        if self.flag_debug:
            self._read_bin_data_debug(bin_data)
        
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
                
                temp_first_4bit_processed_flag = False
                
                for _ in range(temp_data_block["data_size_per_data_block"] - 1):
                    
                    if temp_data_block["sample_size"] == 0:
                        
                        if not temp_first_4bit_processed_flag:
                            temp_byte = bin_data.read(1)
                            temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True) & 0x0F
                            # temp_inc = struct.unpack(">b", temp_byte)[0] & 0x0F
                            temp_first_4bit_processed_flag = True
                        elif temp_first_4bit_processed_flag:
                            temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True) & 0xF0
                            # temp_inc = struct.unpack(">b", temp_byte)[0] & 0xF0
                            temp_first_4bit_processed_flag = False
                    else:   
                        if temp_data_block["sample_size"] == 1:
                            temp_inc_bit = bin_data.read(1)
                        elif temp_data_block["sample_size"] == 2:
                            temp_inc_bit = bin_data.read(2)
                        elif temp_data_block["sample_size"] == 3:
                            temp_inc_bit = bin_data.read(3)
                        elif temp_data_block["sample_size"] == 4:
                            temp_inc_bit = bin_data.read(4)
                        temp_inc = int.from_bytes(temp_inc_bit, byteorder='big', signed=True)
                        
                    temp_sample_value += temp_inc
                    temp_data_block_sample.append(temp_sample_value)
                
                temp_data_block["data"] = temp_data_block_sample
                temp_second_block_data.append(temp_data_block)
            
            temp_second_block["data"] = temp_second_block_data
            temp_data.append(temp_second_block)
        
        # covert data to obspy format 
        temp_stream = Stream()
        
        for i in range(self.num_channel):
            # in order to process data in obspy, the dtype should be int32, not int64
            # see obspy.io.mseed.headers.py line 109
            temp_trace = Trace(data=np.array([temp_data[j]["data"][i]["data"] for j in range(len(temp_data))]).flatten().astype(np.int32))
            temp_trace.stats.sampling_rate = 1 / (temp_data[0]["frame_length"] / 1000)
            temp_trace.stats.delta = 1 / temp_trace.stats.sampling_rate
            temp_trace.stats.calib = self.calib_coeff
            temp_trace.stats.npts = len(temp_trace.data)
            temp_trace.stats.network = ""
            temp_trace.stats.location = ""
            temp_trace.stats.station = ""
            temp_trace.stats.channel = ""
            temp_trace.stats.starttime = UTCDateTime(temp_start_datetime_dt)
            temp_stream.append(temp_trace)
        
        # prepare header
        self.header["start_datatime"] = temp_data[0]["start_datetime"]
        self.header["end_datetime"] = temp_data[-1]["start_datetime"]
        self.header["sampling_rate"] = 1 / (temp_data[0]["frame_length"] / 1000)
        self.header["calib_coeff"] = self.calib_coeff
        self.header["num_channel"] = self.num_channel
        
        return (self.header, temp_stream)
        
    # debugging function
    def _read_bin_data_debug(self, bin_data):
        
        bin_data = io.BytesIO(bin_data) 
        self.header = {}
        self.header["WIN32 Header"] = bin_data.read(4).hex()
        
        target_channel = self.debug_channel
        target_index_second_block = int(self.debug_start_index / 100)
        
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
                    print(temp_start_datetime)
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
                
                temp_first_4bit_processed_flag = False
                
                if i == target_channel and target_index_second_block == len(temp_data):
                    temp_data_block_sample = []
                    temp_sample_value = struct.unpack(">i", bin_data.read(4))[0]
                    temp_data_block_sample.append(temp_sample_value)
                    
                    print("Channel:", i, "Index:", len(temp_data) * 100, 
                          "Sample Size:", temp_data_block["sample_size"], 
                          "Data Size:", temp_data_block["data_size_per_data_block"],
                          "offset", hex(bin_data.tell() + 1024)) 
                    
                    temp_first_4bit_processed_flag = False
                    
                    for j in range(temp_data_block["data_size_per_data_block"] - 1):
                        
                        if temp_data_block["sample_size"] == 0:
                            
                            if not temp_first_4bit_processed_flag:
                                temp_byte = bin_data.read(1)
                                temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True) & 0x0F

                                temp_first_4bit_processed_flag = True
                            elif temp_first_4bit_processed_flag:
                                temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True) & 0xF0
                                temp_first_4bit_processed_flag = False
                                
                        elif temp_data_block["sample_size"] == 1:
                            temp_byte = bin_data.read(1)
                        elif temp_data_block["sample_size"] == 2:
                            temp_byte = bin_data.read(2)
                        elif temp_data_block["sample_size"] == 3:
                            temp_byte = bin_data.read(3)
                        elif temp_data_block["sample_size"] == 4:
                            temp_byte = bin_data.read(4)
                        temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True)
                            
                        print([j, temp_byte, temp_inc])
                    
                    
                    raise ValueError("Debugging is done")
                else:
                    
                    for _ in range(temp_data_block["data_size_per_data_block"] - 1):
                        
                        if temp_data_block["sample_size"] == 0:
                            
                            if not temp_first_4bit_processed_flag:
                                temp_byte = bin_data.read(1)
                                temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True) & 0x0F
                                # temp_inc = struct.unpack(">b", temp_byte)[0] & 0x0F
                                temp_first_4bit_processed_flag = True
                            elif temp_first_4bit_processed_flag:
                                temp_inc = int.from_bytes(temp_byte, byteorder='big', signed=True) & 0xF0
                                # temp_inc = struct.unpack(">b", temp_byte)[0] & 0xF0
                                temp_first_4bit_processed_flag = False
                        else:   
                            if temp_data_block["sample_size"] == 1:
                                temp_inc_bit = bin_data.read(1)
                            elif temp_data_block["sample_size"] == 2:
                                temp_inc_bit = bin_data.read(2)
                            elif temp_data_block["sample_size"] == 3:
                                temp_inc_bit = bin_data.read(3)
                            elif temp_data_block["sample_size"] == 4:
                                temp_inc_bit = bin_data.read(4)
                            temp_inc = int.from_bytes(temp_inc_bit, byteorder='big', signed=True)
                            
                        temp_sample_value += temp_inc
                        temp_data_block_sample.append(temp_sample_value)
                    
                    temp_data_block["data"] = temp_data_block_sample
                    temp_second_block_data.append(temp_data_block)
                
            temp_second_block["data"] = temp_second_block_data
            temp_data.append(temp_second_block)