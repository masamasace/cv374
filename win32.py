from pathlib import Path
import struct
import io
# Win32 data class
# two options to import data
# 1. from the file path: Win32Data(file_path)
# 2. from the binary data: Win32Data.read_bin(bin_data)
# Both options will return the Win32Data object including self.win32_bin_data as binary data

class Win32Data():
    
    def __init__(self, file_path=None, bin_data=None, calib_coeff=1):
        
        if file_path:
            self.header, self.data = self.set_file_path(file_path)
        
        elif bin_data:
            self.header, self.data = self.set_bin_data(bin_data)
        
        self.calib_coeff = calib_coeff
            
    
    def set_file_path(self, file_path):
        self.file_path = Path(file_path).resolve()
        
        with open(self.file_path, "rb") as f:
            temp_bin = f.read()
        
        temp_header, temp_data = self.set_bin_data(temp_bin)
        
        return (temp_header, temp_data)

    
    def set_bin_data(self, bin_data):
        
        bin_data = io.BytesIO(bin_data) 
        temp_header = {}
        temp_header["WIN32 Header"] = bin_data.read(4).hex()
        # discard the first 4 bytes
        
        while True:
            temp_second_block_header = {
                "start_datetime": struct.unpack(">8s", bin_data.read(8))[0].hex(),
                "frame_length": struct.unpack(">I", bin_data.read(4))[0],
                "channel_data_block_length": struct.unpack(">I", bin_data.read(4))[0],
            }
            temp_data = []
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
                    raise ValueError("Not implemented for sample size 0")
                elif temp_data_block["sample_size"] == 1:
                    temp_inc = struct.unpack(">b", bin_data.read(1))[0]
                elif temp_data_block["sample_size"] == 2:
                    temp_inc = struct.unpack(">h", bin_data.read(2))[0]
                elif temp_data_block["sample_size"] == 3:
                    # 24 bit float
                    temp_inc
                elif temp_data_block["sample_size"] == 4:
                    temp_inc = struct.unpack(">i", bin_data.read(4))[0]
                    
            
            
            print(temp_data_block)
            break

        return (temp_header, None)

    
                
    
    
    