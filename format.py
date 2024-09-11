from pathlib import Path
import pandas as pd
import datetime as dt
from .t3w import T3WHandler
from .log import LogHandler
import warnings
import numpy as np

class DataFormatter:
    def __init__(self, data_dir: Path, time_zone: str = "Japan"):
        """
        Parameters
        ----------
        data_dir : Path
            The directory where the microtremor data is stored.
            data_dir should have the following structure:

            data_dir
            ├── sub_dir_1
            │   ├── t3w_file_1_1.t3w
            │   ├── t3w_file_1_2.t3w
            │   ├── ...
            │   ├── log_file_1_1.log
            │   ├── log_file_1_2.log
            │   └── ...
            ├── sub_dir_2
            │   ├── t3w_file_2_1.t3w
            │   ├── t3w_file_2_2.t3w
            │   ├── ...
            │   ├── log_file_2_1.log
        
        time_zone : str
            The time zone of the data. Default is "Japan".
            Please refer to the following link for the list of time zones:
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        """

        self.data_dir = Path(data_dir).resolve()
        self.time_zone = time_zone

        self.export_col_t3w = ["file_path", "rel_file_path", "sub_dir_index", "sub_dir_name", "stem", "group_index", "match_log_index"]
        self.export_col_log = ["file_path", "rel_file_path", "sub_dir_index", "sub_dir_name", "stem"]

        print("-" * 40)
        print(f"Data directory: {self.data_dir}")

        self._create_temp_root_dir()
        self._create_result_root_dir()

        self._create_file_list()

        self._create_temp_sub_dir()
        self._create_result_sub_dir()
        
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
    
    def _create_result_root_dir(self):

        self.res_root_dir = self.data_dir.parent / "res"

        if not self.res_root_dir.exists():
            self.res_root_dir.mkdir(parents=True)
    
    
    def _create_temp_root_dir(self):

        self.tmp_root_dir = self.data_dir.parent / "tmp"

        if not self.tmp_root_dir.exists():
            self.tmp_root_dir.mkdir(parents=True)


    def _create_file_list(self):

        self.log_file_list_path = self.res_root_dir / "log_file_list.csv"
        self.t3w_file_list_path = self.res_root_dir / "t3w_file_list.csv"

        if self.log_file_list_path.exists():
            self.log_file_list, self.t3w_file_list, self.sub_dir_list = self._import_file_list()
        
        else:
            self.log_file_list, self.t3w_file_list, self.sub_dir_list = self._create_file_list_from_scratch()
        
        print(self.t3w_file_list)

    def _import_file_list(self):

        temp_log_file_list = pd.read_csv(self.log_file_list_path, dtype="str")
        temp_t3w_file_list = pd.read_csv(self.t3w_file_list_path, dtype="str")

        # convert dtype
        temp_t3w_file_list[["sub_dir_index", "group_index", "match_log_index"]] = temp_t3w_file_list[["sub_dir_index", "group_index", "match_log_index"]].astype("int")
        temp_log_file_list["sub_dir_index"] = temp_log_file_list["sub_dir_index"].astype("int")


        # if any of the files does not exist, update the file list
        if not "file_path" in temp_log_file_list.columns or \
            temp_log_file_list["file_path"].apply(lambda x: Path(x).exists()).all():
            temp_log_file_list["file_path"] = temp_log_file_list["rel_file_path"].apply(lambda x: self.data_dir / x)
        
            if not temp_log_file_list["file_path"].apply(lambda x: Path(x).exists()).all():
                raise FileNotFoundError("Some of the log files do not exist")
            
        temp_log_file_list["file_path"] = temp_log_file_list["file_path"].apply(lambda x: Path(x))
        
        if not "file_path" in temp_t3w_file_list.columns or \
            not temp_t3w_file_list["file_path"].apply(lambda x: Path(x).exists()).all():
            temp_t3w_file_list["file_path"] = temp_t3w_file_list["rel_file_path"].apply(lambda x: self.data_dir / x)
        
            if not temp_t3w_file_list["file_path"].apply(lambda x: Path(x).exists()).all():
                raise FileNotFoundError("Some of the t3w files do not exist")
        
        temp_t3w_file_list["file_path"] = temp_t3w_file_list["file_path"].apply(lambda x: Path(x))

        # create sub_dir_index column
        temp_sub_dir_list = set(np.concatenate([temp_t3w_file_list["sub_dir_name"].unique(), temp_log_file_list["sub_dir_name"].unique()]))
        temp_sub_dir_list = sorted(temp_sub_dir_list)
        temp_sub_dir_list = pd.DataFrame(temp_sub_dir_list, columns=["sub_dir_name"])
        temp_sub_dir_list["sub_dir_index"] = temp_sub_dir_list.index

        return (temp_log_file_list, temp_t3w_file_list, temp_sub_dir_list)
        

    def _create_file_list_from_scratch(self):
        
        # list t3w_file_list and log_file_list
        temp_t3w_file_list = list(self.data_dir.glob("**/*.t3w"))
        temp_log_file_list = list(self.data_dir.glob("**/*.log"))

        # sort the files
        temp_t3w_file_list = pd.DataFrame(temp_t3w_file_list, columns=["file_path"])
        temp_log_file_list = pd.DataFrame(temp_log_file_list, columns=["file_path"])

        # add stem to the dataframes
        temp_t3w_file_list["stem"] = temp_t3w_file_list["file_path"].apply(lambda x: x.stem)
        temp_log_file_list["stem"] = temp_log_file_list["file_path"].apply(lambda x: x.stem)

        # create sub_dir_index column
        temp_t3w_file_list["sub_dir_name"] = temp_t3w_file_list["file_path"].apply(lambda x: x.parent.relative_to(self.data_dir))
        temp_log_file_list["sub_dir_name"] = temp_log_file_list["file_path"].apply(lambda x: x.parent.relative_to(self.data_dir))

        # create relative path
        temp_t3w_file_list["rel_file_path"] = temp_t3w_file_list["file_path"].apply(lambda x: x.relative_to(self.data_dir))
        temp_log_file_list["rel_file_path"] = temp_log_file_list["file_path"].apply(lambda x: x.relative_to(self.data_dir))

        # create sub_dir_name and sub_dir_index dataframes
        temp_sub_dir_name = np.concatenate([temp_t3w_file_list["sub_dir_name"].unique(), temp_log_file_list["sub_dir_name"].unique()])
        temp_sub_dir_name = set(temp_sub_dir_name)
        temp_sub_dir_name = sorted(temp_sub_dir_name)
        temp_sub_dir_list = pd.DataFrame(temp_sub_dir_name, columns=["sub_dir_name"])
        temp_sub_dir_list["sub_dir_index"] = temp_sub_dir_list.index

        # rcreate sub_dir_index column in t3w_file_list and log_file_list
        temp_t3w_file_list["sub_dir_index"] = temp_t3w_file_list["sub_dir_name"].apply(lambda x: temp_sub_dir_list[temp_sub_dir_list["sub_dir_name"] == x].index[0])
        temp_log_file_list["sub_dir_index"] = temp_log_file_list["sub_dir_name"].apply(lambda x: temp_sub_dir_list[temp_sub_dir_list["sub_dir_name"] == x].index[0])

        # sort the files by sub_dir_index and stem
        temp_t3w_file_list = temp_t3w_file_list.sort_values(by=["sub_dir_index", "stem"])
        temp_log_file_list = temp_log_file_list.sort_values(by=["sub_dir_index", "stem"])
        
        # reset the index
        temp_t3w_file_list = temp_t3w_file_list.reset_index(drop=True)
        temp_log_file_list = temp_log_file_list.reset_index(drop=True)

        return (temp_log_file_list, temp_t3w_file_list, temp_sub_dir_list)


    def _create_temp_sub_dir(self):
        
        for temp_sub_dir in self.sub_dir_list:
                
            temp_temp_dir = self.tmp_root_dir / self.data_dir.name / temp_sub_dir

            if not temp_temp_dir.exists():
                temp_temp_dir.mkdir(parents=True)
        

    def _create_result_sub_dir(self):

        for temp_sub_dir in self.sub_dir_list["sub_dir_name"]:
            
            temp_result_dir = self.res_root_dir / self.data_dir.name / temp_sub_dir

            if not temp_result_dir.exists():
                temp_result_dir.mkdir(parents=True)
        

    # load and store the instance of T3WHandler and LogHandler
    def _load_files(self):
        
        self.log_file_data = self._load_files_log()
        print("Log files are loaded")
        
        self.t3w_file_data = self._load_files_t3w()
        print("T3W files are loaded")

        print("-" * 40) 
    
    
    # load and store the instance of LogHandler
    def _load_files_log(self):

        for i, log_file_path in enumerate(self.log_file_list["file_path"]):
            
            self.log_file_list.loc[i, "data"] = LogHandler(log_file_path)
        

    # load and store the instance of T3WHandler        
    def _load_files_t3w(self):
        
        for i, t3w_file_path in enumerate(self.t3w_file_list["file_path"]):

            self.t3w_file_list.loc[i, "data"] = T3WHandler(t3w_file_path)
    

    # match the t3w and log files
    # make three labels to the t3w files
    # 1. the subdirectory index of the t3w file
    # 2. the index of the log file which is matched with the t3w file
    # 3. the group number of the t3w file

    def _match_files(self):

        self.t3w_file_list["match_log_index"] = -1
        self.t3w_file_list["group_index"] = -1

        temp_group_index = 0

        for i in range(len(self.t3w_file_list)):

            temp_t3w_file_data = self.t3w_file_list.loc[i, "data"]

            # set group number
            # there are two ways to get the start time of the t3w file
            # 1. use the start datetime in the header of the t3w file
            # 2. use the start datetime of the stem of the t3w file
            # the second way is recommended because we have no way to process non-existing files 
            # but both are implemented here
            # the second way is higher priority than the first way
            # TODO: time_zone of start_datetime is hard-coded

            # 1. use the start datetime in the header of the t3w file
            temp_start_datetime_from_t3w = temp_t3w_file_data.header["start_datetime_this_file"]
            temp_start_datetime_from_t3w = dt.datetime.strptime(temp_start_datetime_from_t3w, "%Y%m%d%H%M%S%f")
            temp_start_datetime_from_t3w = temp_start_datetime_from_t3w.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
            temp_start_datetime_first_file_from_t3w = temp_t3w_file_data.header["start_datetime_first_file"]
            temp_start_datetime_first_file_from_t3w = dt.datetime.strptime(temp_start_datetime_first_file_from_t3w,
                                                                                "%Y%m%d%H%M%S%f")
            temp_start_datetime_first_file_from_t3w = temp_start_datetime_first_file_from_t3w.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
            temp_sequnce_number = temp_t3w_file_data.header["sequence_number"]

            # 2. use the start datetime of the stem of the t3w file
            temp_start_datetime_from_t3w = self.t3w_file_list.loc[i, "stem"]
            temp_start_datetime_from_t3w = dt.datetime.strptime(temp_start_datetime_from_t3w[:-4],
                                                                "%Y%m%d%H%M%S")
            temp_start_datetime_from_t3w = temp_start_datetime_from_t3w.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
            temp_recording_duration_from_t3w = temp_t3w_file_data.header["recording_duration"]
            temp_recording_duration_from_t3w = dt.timedelta(seconds=temp_recording_duration_from_t3w)
            temp_end_datetime_from_t3w = temp_start_datetime_from_t3w + temp_recording_duration_from_t3w

            # match the group number
            if i == 0:
                self.t3w_file_list.loc[i, "group_index"] = temp_group_index
            else:

                temp_t3w_file_stem_prev = self.t3w_file_list.loc[i-1, "file_path"].stem
                temp_t3w_file_datetime_prev = dt.datetime.strptime(temp_t3w_file_stem_prev[:-4], "%Y%m%d%H%M%S")
                temp_t3w_file_datetime_prev = temp_t3w_file_datetime_prev.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))
                temp_recording_duration_prev = self.t3w_file_list.loc[i-1, "data"].header["recording_duration"]
                temp_recording_duration_prev = dt.timedelta(seconds=temp_recording_duration_prev)

                if (temp_start_datetime_from_t3w - temp_t3w_file_datetime_prev - temp_recording_duration_prev).total_seconds() != 0:
                    temp_group_index += 1

                # check by using temp_sequnce_number
                # if the group number is different from the previous one
                # the sequence number should be 0
                # otherwise, there is a/some missing files in the directory
                if self.t3w_file_list.loc[i, "group_index"] != self.t3w_file_list.loc[i-1, "group_index"]:
                    if temp_sequnce_number != 0:
                        warnings.warn(f"\nThere may be missing files before {self.t3w_file_list.loc[i, 'file_path']}")
                # the opposite case is not possible

                self.t3w_file_list.loc[i, "group_index"] = temp_group_index
            
            # set the index of the log file which is matched with the t3w file
            for j, log_file_path in enumerate(self.log_file_list["file_path"]):
                
                temp_log_file_data = self.log_file_list.loc[j, "data"]
                temp_start_datetime_from_log = temp_log_file_data.stats["start_time"]
                temp_end_datetime_from_log = temp_log_file_data.stats["end_time"]
                
                if temp_start_datetime_from_log <= temp_end_datetime_from_t3w \
                    and temp_start_datetime_from_t3w <= temp_end_datetime_from_log \
                    and self.t3w_file_list.loc[i, "sub_dir_index"] == self.log_file_list.loc[j, "sub_dir_index"]:
                    
                    self.t3w_file_list.loc[i, "match_log_index"] = j
                    break
        
        self._export_file_list()
    
    def _export_file_list(self, not_update_columns=["group_index"]):
        
        temp_t3w_file_list = self.t3w_file_list.copy()
        temp_t3w_file_list = temp_t3w_file_list[self.export_col_t3w]
        temp_t3w_file_list.to_csv(self.res_root_dir / "t3w_file_list.csv", index=False)

        temp_log_file_list = self.log_file_list.copy()
        temp_log_file_list = temp_log_file_list[self.export_col_log]
        temp_log_file_list.to_csv(self.res_root_dir / "log_file_list.csv", index=False)

        print(self.t3w_file_list)
        print(temp_t3w_file_list)

