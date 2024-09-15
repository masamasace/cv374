from pathlib import Path
import pandas as pd
import datetime as dt
from .t3w import T3WHandler
from .log import LogHandler
import warnings
import numpy as np
import matplotlib.pyplot as plt
import hvsrpy
import gc
from scipy.stats import gmean, gstd

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

def setup_figure(num_row=1, num_col=1, width=5, height=4, left=0.125, 
                 right=0.9, hspace=0.2, wspace=0.2):

    fig, axes = plt.subplots(num_row, num_col, figsize=(width, height), squeeze=False)   
    fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
    return (fig, axes)


class DataFormatter:
    def __init__(self, data_dir: Path, time_zone: str = "Japan",
                 flag_leave_original = {"location": False}):
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

        self.flag_leave_original = flag_leave_original

        print("-" * 40)
        print(f"Data directory: {self.data_dir}")

        self._create_temp_root_dir()
        self._create_result_root_dir()

        self._create_file_list()

        self._create_temp_sub_dir()
        self._create_result_sub_dir()
        
        self._load_files()
        
        # self._check_integrity()
        ## check the integrity of the data
        ## sometimes the t3w files are not continuous or overlapped
        ## while it is not possible to check whether the t3w files are continuous, 
        ## but it is possible to check whether the t3w files are overlapped
        ## if the t3w files are overlapped, the data should not be concatenated
        
        # self._concatenate_t3w_files()
        ## this is too much complicated
        ## 1. validate the unknown overlaps with the end of the previous file and the start of the next file
        ##    -> done by the _check_integrity()    
        ## 2. how to store the data (again create new variables?)
        
        # TODO: add if statement to check whether the columns are in the dataframe
        self._match_files()
        self._marge_log_files(flag_leave_original=self.flag_leave_original["location"])

        # self._create_stationXML()
        
    def _create_result_root_dir(self):

        self.res_root_dir = self.data_dir.parent / "res"

        if not self.res_root_dir.exists():
            self.res_root_dir.mkdir(parents=True)
    
    
    def _create_temp_root_dir(self):

        self.tmp_root_dir = self.data_dir.parent / "tmp"

        if not self.tmp_root_dir.exists():
            self.tmp_root_dir.mkdir(parents=True)


    def _create_file_list(self):

        self.log_file_list_path = self.res_root_dir / self.data_dir.name / "log_file_list.csv"
        self.t3w_file_list_path = self.res_root_dir / self.data_dir.name / "t3w_file_list.csv"

        if self.log_file_list_path.exists():
            self.log_file_list, self.t3w_file_list, self.sub_dir_list = self._import_file_list()
        
        else:
            self.log_file_list, self.t3w_file_list, self.sub_dir_list = self._create_file_list_from_scratch()
        

    def _import_file_list(self):

        temp_log_file_list = pd.read_csv(self.log_file_list_path, dtype="str")
        temp_t3w_file_list = pd.read_csv(self.t3w_file_list_path, dtype="str")

        # convert dtype
        temp_t3w_file_list[["sub_dir_index", "group_index", "match_log_index"]] = temp_t3w_file_list[["sub_dir_index", "group_index", "match_log_index"]].astype("int")
        temp_log_file_list["sub_dir_index"] = temp_log_file_list["sub_dir_index"].astype("int")


        # if any of the files does not exist, update the file list
        if not "file_path" in temp_log_file_list.columns:
            temp_log_file_list["file_path"] = temp_log_file_list["rel_file_path"].apply(lambda x: self.data_dir / x)
        else:
            for i, temp_file_path in enumerate(temp_log_file_list["file_path"]):
                if not Path(temp_file_path).exists():
                    temp_log_file_list.loc[i, "file_path"] = self.data_dir / temp_log_file_list.loc[i, "rel_file_path"]
                    
        temp_log_file_list["file_path"] = temp_log_file_list["file_path"].apply(lambda x: Path(x))
        if not temp_log_file_list["file_path"].apply(lambda x: Path(x)).all():
            raise FileNotFoundError("Some of the log files do not exist")        
            
        
        if not "file_path" in temp_t3w_file_list.columns:
            temp_t3w_file_list["file_path"] = temp_t3w_file_list["rel_file_path"].apply(lambda x: self.data_dir / x)
        else:
            for i, temp_file_path in enumerate(temp_t3w_file_list["file_path"]):
                if not Path(temp_file_path).exists():
                    temp_t3w_file_list.loc[i, "file_path"] = self.data_dir / temp_t3w_file_list.loc[i, "rel_file_path"]
        
        temp_t3w_file_list["file_path"] = temp_t3w_file_list["file_path"].apply(lambda x: Path(x))
        if not temp_t3w_file_list["file_path"].apply(lambda x: Path(x)).all():
            raise FileNotFoundError("Some of the t3w files do not exist")

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
        
        for temp_sub_dir in self.sub_dir_list["sub_dir_name"]:
                
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
        
        self._load_files_log()
        print("Log files are loaded")
        
        self._load_files_t3w()
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
        
    def _export_file_list_csv(self, not_update_columns=["group_index"]):
        
        # TODO: remove the columns which are not important
        
        temp_t3w_file_list = self.t3w_file_list.copy()
        temp_t3w_file_list.to_csv(self.res_root_dir / self.data_dir.name / "t3w_file_list.csv", index=False)

        temp_log_file_list = self.log_file_list.copy()
        temp_log_file_list.to_csv(self.res_root_dir / self.data_dir.name / "log_file_list.csv", index=False)


    def _check_data_conversion(self, ref_dir=None):
        """
        check data conversion by Win32Handler
        Win32Handler.stream is obspy.core.stream.Stream object
        ref_dir contains the reference data converted by PWave32 software
        file name is like 20231018001000.200.dbl.01.asc
            the last 01 is the channel number (01, 02, 03)
        the header of the file is like the following:
            > Station Name        to-soku win
            > Trigger Time        2023/10/18 00:10:00.00
            > Delay Time(s)       0.000
            > Last Corrected Time 2023/10/18 00:06:30
            > Sampling Freq(Hz)   100
            > Duration Time(s)    300.000
            > Channel Name        CH1
            > Unit of Data        cm/s
            > 0.000417
            > 0.000242
            > ...
        """
        
        temp_ref_dir = Path(ref_dir).resolve()
        
        for i in range(len(self.t3w_file_list)):
            
            temp_t3w_data = self.t3w_file_list.loc[i, "data"]
            temp_t3w_path = self.t3w_file_list.loc[i, "file_path"]
            temp_t3w_stem = temp_t3w_path.stem
            
            print(f"Checking the data conversion of {temp_t3w_stem}...", end=" ")
            
            temp_asc_list = list(temp_ref_dir.glob(f"**/*{temp_t3w_stem}*.asc"))
            
            if len(temp_asc_list) == 0:
                print("No reference data found")
                continue
            
            for i in range(len(temp_asc_list)):
                
                temp_asc_channel = int(temp_asc_list[i].stem[-2:]) - 1
                temp_asc_data = pd.read_csv(temp_asc_list[i], skiprows=8, header=None)
                
                temp_t3w_data_channel = temp_t3w_data.stream[temp_asc_channel].data
                temp_t3w_data_calib = temp_t3w_data.calib_coeff
                temp_t3w_data_channel_scaled = temp_t3w_data_channel * temp_t3w_data_calib
                
                temp_diff = temp_t3w_data_channel_scaled - temp_asc_data.values.flatten()
                
                if np.allclose(temp_diff, 0, atol=1e-6):
                    print("Ch.", temp_asc_channel + 1, "OK", end=" ")
                else:
                    print()
                    print(f"Checking the data conversion of {temp_t3w_stem}... ", "Ch.", temp_asc_channel + 1, "NG")
                    
                    temp_diff_index = np.where(np.abs(temp_diff) > 1e-6)[0]
                    print("indices: ", temp_diff_index)
                    print("ref_data_int: ",temp_asc_data.values.flatten()[temp_diff_index])
                    temp_t3w_data_reread = T3WHandler(temp_t3w_path, flag_debug=True, 
                                                      debug_params=[temp_asc_channel, temp_diff_index[0], temp_diff_index[-1]])
                    
                if i == len(temp_asc_list) - 1:
                    print()
        
        
    def _check_integrity(self):

        pass
    
    def _marge_log_files(self, flag_leave_original=False):
        
        if flag_leave_original:
            pass
        
        else:
            for i in range(len(self.t3w_file_list)):
                
                match_log_index = self.t3w_file_list.loc[i, "match_log_index"]
                if match_log_index == -1:
                    continue
                
                temp_log_file_data = self.log_file_list.loc[match_log_index, "data"]
                
                self.t3w_file_list.loc[i, "latitude"] = temp_log_file_data.stats["latitude"]
                self.t3w_file_list.loc[i, "longitude"] = temp_log_file_data.stats["longitude"]
                self.t3w_file_list.loc[i, "altitude"] = temp_log_file_data.stats["altitude"]
                self.t3w_file_list.loc[i, "geoid_height"] = temp_log_file_data.stats["geoid_height"]
                self.t3w_file_list.loc[i, "num_satellites"] = temp_log_file_data.stats["num_satellites"]
                self.t3w_file_list.loc[i, "HDOP"] = temp_log_file_data.stats["HDOP"]

        self._export_file_list_csv()
    
    
    def _concatenate_t3w_files(self):
        
        pass
    
    def _export_stationXML(self):
        
        pass
    
    def export_mseed(self, force_overwrite=False):
        
        mseed_file_list = []
        
        for i in range(len(self.t3w_file_list)):
            temp_t3w_data = self.t3w_file_list.loc[i, "data"]
            temp_mseed_file_path = self.tmp_root_dir / self.data_dir.name / self.t3w_file_list.loc[i, "sub_dir_name"] / (self.t3w_file_list.loc[i, "file_path"].stem + ".mseed")
            
            if force_overwrite or not temp_mseed_file_path.exists():
                temp_t3w_data.stream.write(str(temp_mseed_file_path), format="mseed")
            
            mseed_file_list.append(temp_mseed_file_path)
        
        return mseed_file_list
    
    def calculate_HVSR(self, force_overwrite=False):
        
        self.mseed_file_list = self.export_mseed()
        self.preproc_settings, self.proc_settings = self._create_HVSR_settings()
        
        self.hvsr_list = []
        self.srecords_list = []
        
        for i, mseed_file in enumerate(self.mseed_file_list):
            
            temp_hvsr, temp_srecords = self._calculate_HVSR_base(mseed_file)
        
            self.hvsr_list.append(temp_hvsr)
            self.srecords_list.append(temp_srecords)
            
            if np.isinf(temp_hvsr.amplitude).any() == False and \
                temp_hvsr.mean_curve_peak() is not None and \
                force_overwrite:
                self.t3w_file_list.loc[i, "mean_curve_freq"] = temp_hvsr.mean_curve_peak()[0]
                self.t3w_file_list.loc[i, "mean_curve_amp"] = temp_hvsr.mean_curve_peak()[1]
            
        print("-" * 40)
        print("HVSR calculation is done")
        
        self._export_file_list_csv()
    
    def _create_HVSR_settings(self):
        
        temp_preproc_settings = hvsrpy.settings.HvsrPreProcessingSettings()
        temp_preproc_settings.detrend = "linear"
        temp_preproc_settings.window_length_in_seconds = 40.96
        temp_preproc_settings.orient_to_degrees_from_north = 0.0
        temp_preproc_settings.filter_corner_frequencies_in_hz = (None, None)
        temp_preproc_settings.ignore_dissimilar_time_step_warning = False
        
        temp_proc_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
        temp_proc_settings.window_type_and_width = ("tukey", 0.2)
        temp_proc_settings.smoothing=dict(operator="konno_and_ohmachi",
                                        bandwidth=40,
                                        center_frequencies_in_hz=np.geomspace(0.2, 50, 200))
        temp_proc_settings.method_to_combine_horizontals = "geometric_mean"
        temp_proc_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"
        
        return (temp_preproc_settings, temp_proc_settings)
            
    
    def _calculate_HVSR_base(self, mseed_file):
        
        temp_srecords = hvsrpy.read([[str(mseed_file)]])
        temp_srecords = hvsrpy.preprocess(temp_srecords, self.preproc_settings)
        temp_hvsr = hvsrpy.process(temp_srecords, self.proc_settings)
        
        return (temp_hvsr, temp_srecords)

    
    # analyze the HVSR data considering the group number
    # same group number means the record at the same location
    def merge_HVSR(self):
        
        self.group_list = self._create_group_list()
        
        self._export_merged_HVSR()
        
        self._export_group_list_csv()
    
    
    def _create_group_list(self):
        temp_group_list = pd.DataFrame(self.t3w_file_list["group_index"].unique(), 
                                       columns=["group_index"])
        temp_group_list["sub_dir_index"] = -1
        temp_group_list["num_t3w_files"] = -1
        temp_group_list["start_datetime"] = None
        temp_group_list["end_datetime"] = None
        temp_group_list["latitude"] = None
        temp_group_list["longitude"] = None
        temp_group_list["altitude"] = None
        temp_group_list["geoid_height"] = None
        temp_group_list["best_HDOP"] = None
        
        for i in range(len(temp_group_list)):
            temp_group_index = temp_group_list.loc[i, "group_index"]
            temp_same_group_t3w_file_list = self.t3w_file_list[self.t3w_file_list["group_index"] == temp_group_index]
            
            temp_group_list.loc[i, "sub_dir_index"] = temp_same_group_t3w_file_list["sub_dir_index"].unique()[0]
            temp_group_list.loc[i, "num_t3w_files"] = len(temp_same_group_t3w_file_list)
            temp_group_list.loc[i, "start_datetime"] = temp_same_group_t3w_file_list["data"].apply(lambda x: x.header["start_datetime_this_file"]).min()
            temp_group_list.loc[i, "end_datetime"] = temp_same_group_t3w_file_list["data"].apply(lambda x: x.header["start_datetime_this_file"]).max()

            temp_best_HDOP_index = temp_same_group_t3w_file_list["HDOP"].idxmin()

            if temp_best_HDOP_index is not np.nan:
                temp_group_list.loc[i, "latitude"] = temp_same_group_t3w_file_list.loc[temp_best_HDOP_index, "latitude"]
                temp_group_list.loc[i, "longitude"] = temp_same_group_t3w_file_list.loc[temp_best_HDOP_index, "longitude"]
                temp_group_list.loc[i, "altitude"] = temp_same_group_t3w_file_list.loc[temp_best_HDOP_index, "altitude"]
                temp_group_list.loc[i, "geoid_height"] = temp_same_group_t3w_file_list.loc[temp_best_HDOP_index, "geoid_height"]
                temp_group_list.loc[i, "best_HDOP"] = temp_same_group_t3w_file_list.loc[temp_best_HDOP_index, "HDOP"]
                
        return temp_group_list
    
    def _export_merged_HVSR(self):
        
        for i in range(len(self.group_list)):
            
            temp_group_index = self.group_list.loc[i, "group_index"]
            temp_same_group_t3w_file_list = self.t3w_file_list[self.t3w_file_list["group_index"] == temp_group_index]
            
            temp_fig, temp_axes = setup_figure(num_row=1, num_col=1, width=6, height=4)
            
            temp_freq, temp_amp, temp_amp_geo_mean, \
            temp_amp_geo_mean_plus_std, temp_amp_geo_mean_minus_std, \
            temp_amp_geo_mean_peak_freq, temp_amp_geo_mean_peak \
            = self._calculate_merged_HVSR(temp_same_group_t3w_file_list)
            
            # save geomean peak frequency and amplitude to the group_list
            self.group_list.loc[i, "mean_curve_freq"] = temp_amp_geo_mean_peak_freq
            self.group_list.loc[i, "mean_curve_amp"] = temp_amp_geo_mean_peak
                
            for j in range(len(temp_amp)):
                temp_axes[0, 0].plot(temp_freq, temp_amp[j], color="gray", alpha=0.5, linewidth=0.5)
            
            temp_axes[0, 0].plot(temp_freq, temp_amp_geo_mean, color="k", linewidth=1.5)
            temp_axes[0, 0].fill_between(temp_freq, temp_amp_geo_mean_minus_std,
                                        temp_amp_geo_mean_plus_std, color="gray", alpha=0.5)
            temp_axes[0, 0].plot(temp_amp_geo_mean_peak_freq, temp_amp_geo_mean_peak, "ro", markersize=5,
                                 markeredgewidth=0.5, markeredgecolor="k")
            
            temp_axes[0, 0].set_xscale("log")
            temp_axes[0, 0].set_xlabel("Frequency (Hz)")
            temp_axes[0, 0].set_ylabel("H/V Amplitude")
            temp_axes[0, 0].set_xlim(0.2, 10)
            temp_axes[0, 0].set_ylim(ymin=0)
            
            # remove upper and right spines
            temp_axes[0, 0].spines["top"].set_visible(False)
            temp_axes[0, 0].spines["right"].set_visible(False)
            
            # change tick_label format from 10^-1, 10^0, 10^1 to 0.1, 1, 10
            temp_axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.1f}".format(x)))
            temp_axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.1f}".format(y)))            
            
            temp_fig.savefig(self.res_root_dir / self.data_dir.name / \
                temp_same_group_t3w_file_list["file_path"].iloc[0].relative_to(self.data_dir).parent / \
                f"group_{temp_group_index}.svg", bbox_inches="tight")
            plt.close(temp_fig)
            plt.clf()
            plt.cla()
            gc.collect()
        
        self._export_group_list_csv()
    
    def _calculate_merged_HVSR(self, same_group_t3w_file_list):
        
        temp_amp = []
        
        # extract frequency and amplitude
        for i in range(len(same_group_t3w_file_list)):
            
            temp_hvsr_data = self.hvsr_list[same_group_t3w_file_list.index[i]]
            
            # NOTE: length of temp_freq is considered to be the same among the files
            if i == 0:
                temp_freq = temp_hvsr_data.frequency
            
            for j in range(len(temp_hvsr_data.amplitude)):
                temp_amp.append(temp_hvsr_data.amplitude[j])

        temp_amp = np.array(temp_amp)
        
        # drop the inf and nan values
        temp_amp = temp_amp[~np.isinf(temp_amp).any(axis=1)]
        temp_amp = temp_amp[~np.isnan(temp_amp).any(axis=1)]
        
        # calculate geometric mean and +/- 1 standard deviation of the amplitude
        temp_amp_log = np.log(temp_amp)
        temp_amp_geo_mean = np.exp(np.mean(temp_amp_log, axis=0))
        temp_amp_geo_mean_plus_std = np.exp(np.mean(temp_amp_log, axis=0) + np.std(temp_amp_log, axis=0))
        temp_amp_geo_mean_minus_std = np.exp(np.mean(temp_amp_log, axis=0) - np.std(temp_amp_log, axis=0))
        
        # calculate peak frequency and amplitude in the mean curve
        temp_amp_geo_mean_peak = temp_amp_geo_mean.max()
        temp_amp_geo_mean_peak_index = np.where(temp_amp_geo_mean == temp_amp_geo_mean_peak)[0][0]
        temp_amp_geo_mean_peak_freq = temp_freq[temp_amp_geo_mean_peak_index]
        
        return (temp_freq, temp_amp, temp_amp_geo_mean, 
                temp_amp_geo_mean_plus_std, temp_amp_geo_mean_minus_std, 
                temp_amp_geo_mean_peak_freq, temp_amp_geo_mean_peak)

        
    
    def _export_group_list_csv(self):
        
        temp_group_list = self.group_list.copy()
        temp_group_list.to_csv(self.res_root_dir / self.data_dir.name / "group_list.csv", index=False)