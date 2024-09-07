from pathlib import Path
import pandas as pd
import datetime as dt
from .t3w import T3WHandler
from .log import LogHandler
import sqlite3
import warnings

class DataFormatter:
    def __init__(self, data_dir: Path, time_zone: str = "Japan"):
        """
        Parameters
        ----------
        data_dir : Path
            The directory where the microtremor data is stored.
        time_zone : str
            The time zone of the data. Default is "Japan".
            Please refer to the following link for the list of time zones:
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        """

        self.data_dir = Path(data_dir).resolve()
        self.time_zone = time_zone

        print("-" * 40)
        print(f"Data directory: {self.data_dir}")

        self._count_dirs_files()
        
        self._create_result_dir()
        self._create_temp_dir()
        
        self._load_files()
        
        self._match_files()
        
        # self._check_integrity()
        ## check the integrity of the data
        ## sometimes the t3w files are not continuous or overlapped
        ## while it is not possible to check whether the t3w files are continuous, but it is possible to check whether the t3w files are overlapped
        ## if the t3w files are overlapped, the data should not be concatenated
        
        # self._concatenate_t3w_files()
        ## this is too much complicated
        ## 1. validate the unknown overlaps with the end of the previous file and the start of the next file
        ##    -> done by the _check_integrity()    
        ## 2. how to store the data (again create new variables?)
        
        # self._marge_log_files()
        ## marge the information of log files to self.t3w_file_list
        ## especially, latitude, longitude, and elevation
        ## this variable can be summary of the class
        ## self._marge_log_files()
        ##    self._read_t3w_file_csv() : read the csv file including manually added latitude, longitude, and elevation
        ##                                if the csv file does not exist, create the csv file
        ##                                if the csv file exists, read the csv file and verify the consistency with the self.t3w_file_list
        ##    self._add_lat_long_elevation() : add latitude, longitude, and elevation to self.t3w_file_list
        ##                                if the csv file exists, just check the consistency with the latitude, longitude, and elevation                                  
        ##    self._write_t3w_file_csv() : write the csv file including manually added latitude, longitude, and elevation
        
        # self._convert_t3w_to_miniseed()
        # self._create_stationXML()
        
        # self.compute_HVSR()
        ## compute HVSR from the data with use of https://github.com/jpvantassel/hvsrpy
    

    def _count_dirs_files(self):

        # count the number of subdirectories
        self.sub_dir_list = [x for x in self.data_dir.iterdir() if x.is_dir()]

        # sort the subdirectories
        self.sub_dir_list.sort()

        print(f"Number of subdirectories: {len(self.sub_dir_list)}")
        print("-" * 20)

        # count the number of t3w and log files in each subdirectory
        self.t3w_file_list = []
        self.log_file_list = []

        for sub_dir in self.sub_dir_list:
            t3w_files = list(sub_dir.glob("**/*.t3w"))
            log_files = list(sub_dir.glob("**/*.log"))

            # sort the files
            t3w_files = pd.DataFrame(t3w_files, columns=["file_path"])
            log_files = pd.DataFrame(log_files, columns=["file_path"])

            self.t3w_file_list.append(t3w_files)
            self.log_file_list.append(log_files)
        
            print(f"Number of t3w files in {sub_dir.relative_to(self.data_dir)}: {len(t3w_files)}")
            print(f"Number of log files in {sub_dir.relative_to(self.data_dir)}: {len(log_files)}")
    
    
    def _create_result_dir(self):

        for sub_dir in self.sub_dir_list:

            temp_sub_dir = sub_dir.relative_to(self.data_dir.parent)
            temp_result_dir = self.data_dir.parent / "res" / temp_sub_dir

            if not temp_result_dir.exists():
                temp_result_dir.mkdir(parents=True)
    

    def _create_temp_dir(self):
        
        for sub_dir in self.sub_dir_list:
                
            temp_sub_dir = sub_dir.relative_to(self.data_dir.parent)
            temp_temp_dir = self.data_dir.parent / "tmp" / temp_sub_dir

            if not temp_temp_dir.exists():
                temp_temp_dir.mkdir(parents=True)
    
    # load and store the instance of T3WHandler and LogHandler
    def _load_files(self):
        
        self.log_file_data = self._load_files_log()
        print("-" * 20)
        print("Log files are loaded")
        
        self.t3w_file_data = self._load_files_t3w()
        print("-" * 20)
        print("T3W files are loaded")
    
    
    # load and store the instance of LogHandler
    def _load_files_log(self):
        
        temp_log_file_data = []
        
        for i, log_files in enumerate(self.log_file_list):
            
            temp_log_file_data_each_dir = []
            
            for j, log_file in enumerate(log_files["file_path"]):
                
                temp_log_file_data_each = LogHandler(log_file, time_zone=self.time_zone)
                temp_log_file_data_each_dir.append(temp_log_file_data_each)
            
            temp_log_file_data.append(temp_log_file_data_each_dir)
            
        return temp_log_file_data
    
        
    def _load_files_t3w(self):
        
        temp_t3w_file_data = []
        
        for i, t3w_files in enumerate(self.t3w_file_list):
            
            temp_t3w_file_data_each_dir = []
            
            for j, t3w_file in enumerate(t3w_files["file_path"]):
                
                temp_t3w_file_data_each = T3WHandler(t3w_file)
                temp_t3w_file_data_each_dir.append(temp_t3w_file_data_each)
            
            temp_t3w_file_data.append(temp_t3w_file_data_each_dir)
        
        return temp_t3w_file_data
    
    # print lat and long in t3w files
    # this is for checking the data
    # the latitude and longitude in t3w files are not recommended to use
    # because they are not accurate
    def _print_lat_long_from_t3w(self):
        
        for i, t3w_files in enumerate(self.t3w_file_list):
            for j, t3w_file in enumerate(t3w_files["file_path"]):
                
                temp_t3w_file_data = self.t3w_file_data[i][j]
                
                print(t3w_file.stem, end=": ")
                print("latitude: ", temp_t3w_file_data.header["latitude"], end=", ")
                print("longitude: ", temp_t3w_file_data.header["longitude"])
    
    # match the t3w and log files
    # make three labels to the t3w files
    # 1. the subdirectory index of the t3w file
    # 2. the index of the log file which is matched with the t3w file
    # 3. the group number of the t3w file
    
    def _match_files(self):
        
        for i, t3w_files in enumerate(self.t3w_file_list):
            temp_group_number = 0
            
            t3w_files["dir_index"] = -1
            t3w_files["match_log_index"] = -1
            t3w_files["group_number"] = -1
            
            self.log_file_list[i]["dir_index"] = -1
            
            for j, t3w_file in enumerate(t3w_files["file_path"]):
                
                t3w_files.loc[j, "dir_index"] = i
                temp_t3w_file_data = self.t3w_file_data[i][j]
                
                # set group number
                # there are two ways to get the start time of the t3w file
                # 1. use the start datetime in the header of the t3w file
                # 2. use the start datetime of the stem of the t3w file
                # the second way is recommended because we have no way to process non-existing files 
                # but both are implemented here
                # the second way is higher priority than the first way
                # TODO: time_zone of start_datetime is hard-coded
                
                # 1. use the start datetime in the header of the t3w file
                temp_start_datetime_from_header = temp_t3w_file_data.header["start_datetime_this_file"]
                temp_start_datetime_from_header = dt.datetime.strptime(temp_start_datetime_from_header, "%Y%m%d%H%M%S%f")
                temp_start_datetime_from_header = temp_start_datetime_from_header.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
                temp_start_datetime_first_file_from_header = temp_t3w_file_data.header["start_datetime_first_file"]
                temp_start_datetime_first_file_from_header = dt.datetime.strptime(temp_start_datetime_first_file_from_header, 
                                                                                  "%Y%m%d%H%M%S%f")
                temp_start_datetime_first_file_from_header = temp_start_datetime_first_file_from_header.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
                temp_sequnce_number = temp_t3w_file_data.header["sequence_number"]
                
                # 2. use the start datetime of the stem of the t3w file
                temp_start_datetime_from_stem = t3w_file.stem
                temp_start_datetime_from_stem = dt.datetime.strptime(temp_start_datetime_from_stem[:-4], 
                                                                     "%Y%m%d%H%M%S")
                temp_start_datetime_from_stem = temp_start_datetime_from_stem.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
                temp_recording_duration_from_header = temp_t3w_file_data.header["recording_duration"]
                temp_recording_duration_from_header = dt.timedelta(seconds=temp_recording_duration_from_header)
                temp_end_datetime_from_stem = temp_start_datetime_from_stem + temp_recording_duration_from_header
                
                if j == 0:
                    t3w_files.loc[j, "group_number"] = temp_group_number
                else:
                    temp_t3w_file_stem_prev_from_stem = t3w_files.iloc[j-1]["file_path"].stem
                    temp_t3w_file_datetime_prev_from_stem = dt.datetime.strptime(temp_t3w_file_stem_prev_from_stem[:-4], "%Y%m%d%H%M%S")
                    temp_t3w_file_datetime_prev_from_stem = temp_t3w_file_datetime_prev_from_stem.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
                    temp_recording_duration_prev_from_header = self.t3w_file_data[i][j-1].header["recording_duration"]
                    temp_recording_duration_prev_from_header = dt.timedelta(seconds=temp_recording_duration_prev_from_header)
                    
                    if (temp_start_datetime_from_stem - temp_t3w_file_datetime_prev_from_stem - temp_recording_duration_prev_from_header).total_seconds() != 0:
                        temp_group_number += 1
                    
                    t3w_files.loc[j, "group_number"] = temp_group_number
                    
                    # check by using temp_sequnce_number
                    # if the group number is different from the previous one
                    # the sequence number should be 0
                    # otherwise, there is a/some missing files in the directory
                    if t3w_files.loc[j, "group_number"] != t3w_files.loc[j-1, "group_number"]:
                        if temp_sequnce_number != 0:
                            warnings.warn(f"\nThere may be missing files before {t3w_file}")
                    # the opposite case is not possible
                
                # set the index of the log file which is matched with the t3w file
                # TODO: 3-times nested for loop
                
                for k, log_file in enumerate(self.log_file_list[i]["file_path"]):
                    
                    self.log_file_list[i].loc[k, "dir_index"] = i
                    
                    temp_log_file_data = self.log_file_data[i][k]
                    temp_start_datetime_from_log = temp_log_file_data.stats["start_time"]
                    temp_end_datetime_from_log = temp_log_file_data.stats["end_time"]
                    
                    if temp_start_datetime_from_log <= temp_end_datetime_from_stem and temp_start_datetime_from_stem <= temp_end_datetime_from_log:
                        t3w_files.loc[j, "match_log_index"] = k
                        break
            
        

    def _convert_t3w_to_miniseed(self):

        for t3w_files in self.t3w_file_list:
            for t3w_file in t3w_files["file_path"]:

                t3w_data = T3WHandler(t3w_file)
                temp_sub_dir = t3w_file.parent.relative_to(self.data_dir.parent)
                # TODO: to be updated to handle obspy stream
                t3w_data.export_data_mseed(dir_path=self.data_dir.parent / "res" / temp_sub_dir)
    
    
    def _create_stationXML(self):
        
        pass
    