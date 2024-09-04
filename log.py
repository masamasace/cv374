from pathlib import Path
import pandas as pd
import datetime as dt

class LogHandler:
    
    def __init__(self, file_path, time_zone="Japan"):
        
        self.file_path = Path(file_path).resolve()
        self.time_zone = time_zone
        
        self.col_name_GPGGA = ["log_type", "UTC_time", "latitude", "NS",
                               "longitude", "EW", "quality", "num_satellites",
                               "HDOP", "altitude", "M", "geoid_height", "M", "id", "checksum"]
        self.col_name_GPGGA_inuse = ["UTC_time", "local_time", "JP_time", "latitude", "longitude", 
                                     "quality", "num_satellites", "HDOP", "altitude", "geoid_height"]
        self.col_name_GPZDA = ["log_type", "UTC_time", "day", "month", "year", "local_zone", "local_zone_minute-checksum"]
        
        
        self._read_log_data_raw()
        self._process_log_data()
    
    def _read_log_data_raw(self):
        
        with open(self.file_path, "r") as f:
            lines = f.readlines()
        
        temp_line_GPGGA = []
        temp_line_GPZDA = []
        
        for line in lines:
            if "$GPGGA" in line:
                temp_line_GPGGA.append(line.strip().split(","))
            elif "$GPZDA" in line:
                temp_line_GPZDA.append(line.strip().split(","))
        
        self.raw_GPGGA = pd.DataFrame(temp_line_GPGGA, columns=self.col_name_GPGGA)
        self.raw_GPZDA = pd.DataFrame(temp_line_GPZDA, columns=self.col_name_GPZDA)
        
        self.raw_GPGGA = self.raw_GPGGA.drop("id", axis=1)
        
        self.raw_GPGGA = self.raw_GPGGA.replace("", pd.NA)
        self.raw_GPZDA = self.raw_GPZDA.replace("", pd.NA)
        self.raw_GPGGA = self.raw_GPGGA.dropna(axis="index").reset_index(drop=True)
        self.raw_GPZDA = self.raw_GPZDA.dropna(axis="index").reset_index(drop=True)
        
        temp_start_date = self.raw_GPZDA["year"][0] + self.raw_GPZDA["month"][0] + self.raw_GPZDA["day"][0]
        temp_start_time= self.raw_GPGGA["UTC_time"][0]
        temp_start_datetime = dt.datetime.strptime(temp_start_date + temp_start_time, "%Y%m%d%H%M%S.%f")
        
        self.raw_GPGGA["UTC_time"] = pd.to_datetime(self.raw_GPGGA["UTC_time"], format="%H%M%S.%f", utc=True)
        self.raw_GPGGA["UTC_time"] = self.raw_GPGGA["UTC_time"].apply(
            lambda x: x.replace(year=temp_start_datetime.year, month=temp_start_datetime.month, day=temp_start_datetime.day))

        self.raw_GPGGA["local_time"] = self.raw_GPGGA["UTC_time"].dt.tz_convert(self.time_zone)
        
        self.raw_GPGGA["JP_time"] = self.raw_GPGGA["UTC_time"].dt.tz_convert("Japan")
        
        self.raw_GPGGA["latitude"] = self.raw_GPGGA["latitude"].astype(float)
        self.raw_GPGGA["longitude"] = self.raw_GPGGA["longitude"].astype(float)
        
        self.raw_GPGGA["latitude"] = self.raw_GPGGA["latitude"].apply(lambda x: int(x/100) + (x % 100)/60)
        self.raw_GPGGA["longitude"] = self.raw_GPGGA["longitude"].apply(lambda x: int(x/100) + (x % 100)/60)
        
        self.raw_GPGGA["latitude"] = self.raw_GPGGA["latitude"] * (1 if self.raw_GPGGA["NS"][0] == "N" else -1)
        self.raw_GPGGA["longitude"] = self.raw_GPGGA["longitude"] * (1 if self.raw_GPGGA["EW"][0] == "E" else -1)
        
        self.raw_GPGGA["quality"] = self.raw_GPGGA["quality"].astype(float)
        self.raw_GPGGA["num_satellites"] = self.raw_GPGGA["num_satellites"].astype(float)
        self.raw_GPGGA["HDOP"] = self.raw_GPGGA["HDOP"].astype(float)
        self.raw_GPGGA["altitude"] = self.raw_GPGGA["altitude"].astype(float)
        self.raw_GPGGA["geoid_height"] = self.raw_GPGGA["geoid_height"].astype(float)
    
    def _process_log_data(self):

        # drop unnecessary columns
        self.raw_GPGGA = self.raw_GPGGA[self.col_name_GPGGA_inuse]

        # filter the data with quality == 1
        temp_raw_GPGGA_filtered = self.raw_GPGGA[self.raw_GPGGA["quality"] >= 1]

        # filter the data with the minimum HDOP
        temp_raw_GPGGA_filtered = temp_raw_GPGGA_filtered[temp_raw_GPGGA_filtered["HDOP"] == temp_raw_GPGGA_filtered["HDOP"].min()]
        
        # take average of the data
        temp_raw_GPGGA_filtered = temp_raw_GPGGA_filtered[["latitude", "longitude", "quality", "num_satellites", "HDOP", "altitude", "geoid_height"]].mean()
        
        # add the data to the log files
        self.stats = {
            "latitude": temp_raw_GPGGA_filtered["latitude"],
            "longitude": temp_raw_GPGGA_filtered["longitude"],
            "quality": temp_raw_GPGGA_filtered["quality"],
            "num_satellites": temp_raw_GPGGA_filtered["num_satellites"],
            "HDOP": temp_raw_GPGGA_filtered["HDOP"],
            "altitude": temp_raw_GPGGA_filtered["altitude"],
            "geoid_height": temp_raw_GPGGA_filtered["geoid_height"],
            "start_time": self.raw_GPGGA["UTC_time"].iloc[0], 
            "end_time": self.raw_GPGGA["UTC_time"].iloc[-1],
        }
        