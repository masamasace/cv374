from pathlib import Path
import pandas as pd
import datetime as dt
import struct
from .t3w import t3wData

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
        self._match_files()
        self._convert_t3w_to_miniseed()
    

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
        

    def _match_files(self):

        print("-" * 20)

        temp_col_name_GPGGA = ["log_type", "UTC_time", "latitude", "NS",
                               "longitude", "EW", "quality", "num_satellites",
                               "HDOP", "altitude", "M", "geoid_height", "M", "id", "checksum"]
        temp_col_name_GPGGA_inuse = ["UTC_time", "local_time", "JP_time", "latitude", "longitude", "quality", "num_satellites", "HDOP", "altitude", "geoid_height"]
        temp_col_name_GPZDA = ["log_type", "UTC_time", "day", "month", "year", "local_zone", "local_zone_minute-checksum"]

        # read the log file which is nmea log file
        for i, log_files in enumerate(self.log_file_list):

            self.t3w_file_list[i]["match_log_index"] = -1
            
            for j, log_file in enumerate(log_files["file_path"]):

                temp_line_GPGGA = []
                temp_line_GPZDA = []

                with open(log_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    if "$GPGGA" in line:
                        temp_line_GPGGA.append(line.strip().split(","))
                    elif "$GPZDA" in line:
                        temp_line_GPZDA.append(line.strip().split(","))
                
                temp_df_GPGGA = pd.DataFrame(temp_line_GPGGA, columns=temp_col_name_GPGGA)
                temp_df_GPZDA = pd.DataFrame(temp_line_GPZDA, columns=temp_col_name_GPZDA)

                # drop id column in GPGGA
                temp_df_GPGGA = temp_df_GPGGA.drop("id", axis=1)

                # drop null rows and reset index
                temp_df_GPGGA = temp_df_GPGGA.replace("", pd.NA)
                temp_df_GPZDA = temp_df_GPZDA.replace("", pd.NA)
                temp_df_GPGGA = temp_df_GPGGA.dropna(axis="index").reset_index(drop=True)
                temp_df_GPZDA = temp_df_GPZDA.dropna(axis="index").reset_index(drop=True)

                temp_start_date = temp_df_GPZDA["year"][0] + temp_df_GPZDA["month"][0] + temp_df_GPZDA["day"][0]
                temp_start_time = temp_df_GPGGA["UTC_time"][0]
                temp_start_datetime = dt.datetime.strptime(temp_start_date + temp_start_time, "%Y%m%d%H%M%S.%f")

                # convert datetime string in GPGGA to dataetime object with use of temp_start_datetime
                temp_df_GPGGA["UTC_time"] = pd.to_datetime(temp_df_GPGGA["UTC_time"], format="%H%M%S.%f", utc=True)
                temp_df_GPGGA["UTC_time"] = temp_df_GPGGA["UTC_time"].apply(lambda x: x.replace(year=temp_start_datetime.year, month=temp_start_datetime.month, day=temp_start_datetime.day))
                
                # make column of local time zone
                temp_df_GPGGA["local_time"] = temp_df_GPGGA["UTC_time"].dt.tz_convert(self.time_zone)

                # make column of Japanese time zone
                temp_df_GPGGA["JP_time"] = temp_df_GPGGA["UTC_time"].dt.tz_convert("Japan")

                # convert latitude and longitude to float
                # ddmm.mmmmmmm -> dd + mm.mmmmmmm/60
                temp_df_GPGGA["latitude"] = temp_df_GPGGA["latitude"].astype(float)
                temp_df_GPGGA["longitude"] = temp_df_GPGGA["longitude"].astype(float)

                temp_df_GPGGA["latitude"] = temp_df_GPGGA["latitude"].apply(lambda x: int(x/100) + (x % 100)/60)
                temp_df_GPGGA["longitude"] = temp_df_GPGGA["longitude"].apply(lambda x: int(x/100) + (x % 100)/60)

                # consider the direction of latitude and longitude
                temp_df_GPGGA["latitude"] = temp_df_GPGGA["latitude"] * (1 if temp_df_GPGGA["NS"][0] == "N" else -1)
                temp_df_GPGGA["longitude"] = temp_df_GPGGA["longitude"] * (1 if temp_df_GPGGA["EW"][0] == "E" else -1)

                # convert quality, num_satellites, HDOP, altitude, geoid_height to float
                temp_df_GPGGA["quality"] = temp_df_GPGGA["quality"].astype(float)
                temp_df_GPGGA["num_satellites"] = temp_df_GPGGA["num_satellites"].astype(float)
                temp_df_GPGGA["HDOP"] = temp_df_GPGGA["HDOP"].astype(float)
                temp_df_GPGGA["altitude"] = temp_df_GPGGA["altitude"].astype(float)
                temp_df_GPGGA["geoid_height"] = temp_df_GPGGA["geoid_height"].astype(float)

                # drop unnecessary columns
                temp_df_GPGGA = temp_df_GPGGA[temp_col_name_GPGGA_inuse]

                # filter the data with quality == 1
                temp_df_GPGGA_filtered = temp_df_GPGGA[temp_df_GPGGA["quality"] >= 1]

                # filter the data with the minimum HDOP
                temp_df_GPGGA_filtered = temp_df_GPGGA_filtered[temp_df_GPGGA_filtered["HDOP"] == temp_df_GPGGA_filtered["HDOP"].min()]
                
                # take average of the data
                temp_df_GPGGA_filtered = temp_df_GPGGA_filtered[["latitude", "longitude", "quality", "num_satellites", "HDOP", "altitude", "geoid_height"]].mean()
                
                # add the data to the log files
                log_files.loc[j, "latitude"] = temp_df_GPGGA_filtered["latitude"]
                log_files.loc[j, "longitude"] = temp_df_GPGGA_filtered["longitude"]
                log_files.loc[j, "quality"] = temp_df_GPGGA_filtered["quality"]
                log_files.loc[j, "num_satellites"] = temp_df_GPGGA_filtered["num_satellites"]
                log_files.loc[j, "HDOP"] = temp_df_GPGGA_filtered["HDOP"]
                log_files.loc[j, "altitude"] = temp_df_GPGGA_filtered["altitude"]
                log_files.loc[j, "geoid_height"] = temp_df_GPGGA_filtered["geoid_height"]

                # check the start and end datetime of the log file
                # this is because stem of t3w file is consisted of the JP time
                temp_start_datetime = temp_df_GPGGA["JP_time"][0]
                temp_end_datetime = temp_df_GPGGA["JP_time"].iloc[-1]

                # find the t3w files which are in the time range of the log file
                temp_group_number = 0

                for k, t3w_file in enumerate(self.t3w_file_list[i]["file_path"]):
                    temp_t3w_file_stem = t3w_file.stem
                    temp_t3w_file_datetime = dt.datetime.strptime(temp_t3w_file_stem[:-4], "%Y%m%d%H%M%S")
                    temp_t3w_file_datetime = temp_t3w_file_datetime.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))

                    if temp_start_datetime <= temp_t3w_file_datetime <= temp_end_datetime:
                        self.t3w_file_list[i].loc[k, "match_log_index"] = j
                    
                    if k == 0:
                        temp_group_number = 0
                    else:
                        temp_t3w_file_stem_prev = self.t3w_file_list[i].iloc[k-1]["file_path"].stem
                        temp_t3w_file_datetime_prev = dt.datetime.strptime(temp_t3w_file_stem_prev[:-4], "%Y%m%d%H%M%S")
                        temp_t3w_file_datetime_prev = temp_t3w_file_datetime_prev.astimezone(tz=dt.timezone(dt.timedelta(hours=9)))

                        if (temp_t3w_file_datetime - temp_t3w_file_datetime_prev).total_seconds() != 60 * 5:
                            temp_group_number += 1
                    
                    self.t3w_file_list[i].loc[k, "group_number"] = temp_group_number
            
            # count unmatched t3w files
            temp_unmatched_t3w_files = self.t3w_file_list[i][self.t3w_file_list[i]["match_log_index"] == -1]
            print(f"Number of unmatched t3w files in {self.sub_dir_list[i].relative_to(self.data_dir)}: {len(temp_unmatched_t3w_files)}")

            # export the log files and t3w files as csv format
            temp_sub_dir = self.sub_dir_list[i].relative_to(self.data_dir.parent)
            temp_log_file = self.data_dir.parent / "res" / temp_sub_dir / "log.csv"
            temp_t3w_file = self.data_dir.parent / "res" / temp_sub_dir / "t3w.csv"

            log_files.to_csv(temp_log_file, index=True, index_label="index")
            self.t3w_file_list[i].to_csv(temp_t3w_file, index=True, index_label="index")


    def _convert_t3w_to_miniseed(self):

        for t3w_files in self.t3w_file_list:
            for t3w_file in t3w_files["file_path"]:

                t3w_data = t3wData(t3w_file)
                temp_sub_dir = t3w_file.parent.relative_to(self.data_dir.parent)
                t3w_data.export_data_mseed(dir_path=self.data_dir.parent / "res" / temp_sub_dir)
                
    