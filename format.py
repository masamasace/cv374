from pathlib import Path
import pandas as pd
import datetime as dt
from .t3w import T3WHandler
from .log import LogHandler
import warnings
import numpy as np
import matplotlib.pyplot as plt
import hvsrpy

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

def setup_figure(num_row=1, num_col=1, width=5, height=4, left=0.125, right=0.9, hspace=0.2, wspace=0.2):

    fig, axes = plt.subplots(num_row, num_col, figsize=(width, height), squeeze=False)   
    fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
    return (fig, axes)


class DataFormatter:
    def __init__(self, data_dir: Path, time_zone: str = "Japan", 
                 keep_original_col=["group_index", "match_log_index", "latitude", 
                                     "longitude", "elevation", "geoid_height"]):
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

        self.export_col_t3w = ["file_path", "rel_file_path", "sub_dir_index", "sub_dir_name", "stem", 
                               "group_index", "match_log_index", "latitude", "longitude", "elevation", "geoid_height"]
        self.export_col_log = ["file_path", "rel_file_path", "sub_dir_index", "sub_dir_name", "stem"]
        self.keep_original_col = keep_original_col

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
        ## while it is not possible to check whether the t3w files are continuous, but it is possible to check whether the t3w files are overlapped
        ## if the t3w files are overlapped, the data should not be concatenated
        
        # self._concatenate_t3w_files()
        ## this is too much complicated
        ## 1. validate the unknown overlaps with the end of the previous file and the start of the next file
        ##    -> done by the _check_integrity()    
        ## 2. how to store the data (again create new variables?)
        
        # if both of ["group_index", "match_log_index"] are in keep_original_col, 
        # the following code will not update the columns
        if not all([col in self.keep_original_col for col in ["group_index", "match_log_index"]]):
            self._match_files()
    
        if not all([col in self.keep_original_col for col in ["latitude", "longitude", "elevation", "geoid_height"]]):    
            self._marge_log_files()


        # self._convert_t3w_to_miniseed()
        # self._create_stationXML()
        
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
        
    def _export_file_list(self, not_update_columns=["group_index"]):
        
        temp_t3w_file_list = self.t3w_file_list.copy()
        temp_t3w_file_list = temp_t3w_file_list[self.export_col_t3w]
        temp_t3w_file_list.to_csv(self.res_root_dir / "t3w_file_list.csv", index=False)

        temp_log_file_list = self.log_file_list.copy()
        temp_log_file_list = temp_log_file_list[self.export_col_log]
        temp_log_file_list.to_csv(self.res_root_dir / "log_file_list.csv", index=False)


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
    
    def _marge_log_files(self):
        
        for i in range(len(self.t3w_file_list)):
            
            match_log_index = self.t3w_file_list.loc[i, "match_log_index"]
            if match_log_index == -1:
                continue
            temp_log_file_data = self.log_file_list.loc[match_log_index, "data"]
            
            self.t3w_file_list.loc[i, "latitude"] = temp_log_file_data.stats["latitude"]
            self.t3w_file_list.loc[i, "longitude"] = temp_log_file_data.stats["longitude"]
            self.t3w_file_list.loc[i, "elevation"] = temp_log_file_data.stats["altitude"]
            self.t3w_file_list.loc[i, "geoid_height"] = temp_log_file_data.stats["geoid_height"]

        self._export_file_list()
    
    
    def _concatenate_t3w_files(self):
        
        pass
    
    def _export_stationXML(self):
        
        pass
    
    def export_sac(self, force_overwrite=True):
        
        sac_file_list = []
        
        for i in range(len(self.t3w_file_list)):
            temp_t3w_data = self.t3w_file_list.loc[i, "data"]
            temp_sac_file_path = self.tmp_root_dir / self.data_dir.name / self.t3w_file_list.loc[i, "sub_dir_name"] / (self.t3w_file_list.loc[i, "file_path"].stem + ".sac")
            
            if force_overwrite or not temp_sac_file_path.exists():
                temp_t3w_data.stream.write(str(temp_sac_file_path), format="SAC")
            
            sac_file_list.append(temp_sac_file_path)
            
        
        return sac_file_list
    
    def calculate_HVSR(self):
        
        # https://github.com/jpvantassel/hvsrpy
        
        self.sac_file_list = self.export_sac()
        
        for sac_file in self.sac_file_list:
            
            self._calculate_HVSR_base(sac_file)
            
            raise NotImplementedError("The following code is not implemented yet")
            
    
    def _calculate_HVSR_base(self, sac_file):
        
        preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
        preprocessing_settings.detrend = "linear"
        preprocessing_settings.window_length_in_seconds = 100
        preprocessing_settings.orient_to_degrees_from_north = 0.0
        preprocessing_settings.filter_corner_frequencies_in_hz = (None, None)
        preprocessing_settings.ignore_dissimilar_time_step_warning = False
        
        preprocessing_settings.psummary()
        
        processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
        processing_settings.window_type_and_width = ("tukey", 0.2)
        processing_settings.smoothing=dict(operator="konno_and_ohmachi",
                                        bandwidth=40,
                                        center_frequencies_in_hz=np.geomspace(0.2, 50, 200))
        processing_settings.method_to_combine_horizontals = "geometric_mean"
        processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

        print("Processing Summary")
        print("-"*60)
        processing_settings.psummary()
        
        srecords = hvsrpy.read([[str(sac_file)]])
        srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
        hvsr = hvsrpy.process(srecords, processing_settings)
        
        print("\nStatistical Summary:")
        print("-"*20)
        hvsrpy.summarize_hvsr_statistics(hvsr)
        (fig, ax) = hvsrpy.plot_single_panel_hvsr_curves(hvsr,)
        ax.get_legend().remove()
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
            
    