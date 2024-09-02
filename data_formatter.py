from pathlib import Path
import pandas as pd
import datetime as dt
import struct

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

                self._convert_t3w_to_miniseed_single(t3w_file)
                
    
    def _convert_t3w_to_miniseed_single(t3w_file: Path):
        
        # 先頭からの位置   型  サイズ    内容
        # ------------------------------------
        # 0    3    -    4  システム・ワーキング
        # 4   15    A    4  装置使用プログラム名
        # 16  23    A    8  装置使用プログラム作成日
        # 24  25    B    2  機番
        # 26  27    B    2  ファイルの大きさ (KByte)
        # 28  29    B    2  ヘッダの大きさ (Byte)
        # 30  31    B    2  記録チャンネル総数
        # 32  35    B    4  波形データサンプリング数 (チャンネル当たり)
        # 36  37    B    2  データ長 (Byte)
        # 38  39    B    2  AD変換ビット数
        # 40  41    B    2  サンプリング時間間隔 (ms)
        # 42  43    B    2  本記録の遅延時間 (秒)
        # 44  47    -    -  拡張用
        # 48  49    B    2  起動条件 (1:自動起動)
        # 50  51    B    2  連続記録における順序 (先頭は1)
        # 52  65    B   14  本記録開始の年・月・日・時・分・秒・1/1000 秒 (各2Byte)
        # 66  79    B   14  先頭記録開始の年・月・日・時・分・秒・1/1000 秒 (各2Byte)
        # 80  93    B   14  最新時刻修正時の年・月・日・時・分・秒・1/1000 秒 (各2Byte)
        # 94  95    B    2  時刻修正要因:1=GPS,2=NTP,3=手動
        # 96 109    B   14  前回シャットダウン時刻の年・月・日・時・分・秒・1/100 秒 (各2Byte)
        # 110 123   -   14  拡張用
        # 124 127   -    4  未使用
        # 128 129   B    2  記録チャンネル総数
        # 130 131   B    2  サンプリング周波数 (Hz)
        # 132 133   B    2  自動起動時の遅延時間 (秒)
        # 134 135   B    2  第1トリガ判断チャンネル (1固定)
        # 136 137   B    2  第1トリガ判断レベル (mkine)
        # 138 139   B    2  第2トリガ判断チャンネル (2固定)
        # 140 141   B    2  第2トリガ判断レベル (mkine)
        # 142 143   B    2  第3トリガ判断チャンネル (3固定)
        # 144 145   B    2  第3トリガ判断レベル (mkine)
        # 146 147   B    2  トリガ論理 (1=or, 2=and, 3=and, 4=off)
        # 148 149   B    2  記録先保存モード (0=最新)
        # 150 151   B    2  第1小判断チャンネル (1固定)
        # 152 153   B    2  第2小判断チャンネル (2固定)
        # 154 155   B    2  第3小判断チャンネル (3固定)
        # 156 157   B    2  トリガ引き延ばし時間 (秒)
        # 158 159   B    2  トリガ終了判断時間長 (秒)
        # 160 161   B    2  未使用
        # 162 191   -   32  メモ書き
        # 192 223   A   32  メモ書き
        # 224 227   B    4  第1チャンネル・オフセット
        # 228 231   B    4  第2チャンネル・オフセット
        # 232 235   B    4  第3チャンネル・オフセット
        # 236 239   -    4  拡張用
        # 240 247   B    8  未使用
        # 248 255   B    8  未使用
        # 256 511   B  256  未使用
        # 512 639   B  128  本記録ファイル内での各チャンネルの最大値の10倍値 (mkine/各2Byte)
        # 640 767   B  128  先頭記録から通しての各チャンネルの最大値の10倍値 (mkine/各2Byte)
        # 768 769   B    2  未使用
        # 770 771   B    2  未使用
        # 772 773   B    2  未使用
        # 774 775   B    2  未使用
        # 776 777   B    2  未使用
        # 778 779   B    2  未使用
        # 780 783   B    4  未使用
        # 784 787   B    4  未使用
        # 788 789   B    2  未使用
        # 790 791   B    2  未使用
        # 792 793   B    2  未使用
        # 794 795   B    2  未使用
        # 796 799   B    4  未使用
        # 800 801   B    2  未使用
        # 802 807   B    6  未使用
        # 808 811   B    4  緯度 (度)
        # 812 815   B    4  緯度 (分)
        # 816 819   B    4  緯度 (度)
        # 820 823   B    4  緯度 (分)
        # 824 827   B    4  未使用
        # 828 828   A    1  北/南極フラグ
        # 829 829   A    1  東/西極フラグ
        # 830 830   B    1  使用衛星数
        # 831 831   B    1  未使用
        # 832 833   B    2  STA/LTAトリガ 0:しない, 1:する
        # 834 835   B    2  STA (Short Term Average: 短時間平均) を計算するウィンドウ長 (秒)
        # 836 837   B    2  LTA (Long Term Average: 長時間平均) を計算するウィンドウ長 (秒)
        # 838 839   B    2  トリガ ON 判定 STA/LTA 比の 10 倍値
        # 840 841   B    2  トリガ OFF 判定 STA/LTA 比の 10 倍値
        # 842 843   B    2  トリガ論理 1: or, 2: and, 3: and, 4: off
        # 844 845   B    2  トリガ ON と判定するまでのサンプリング回数
        # 846 857   B   12  未使用
        # 858 869   B   12  未使用
        # 870 1023  -  154  未使用
        # ------------------------------------
        # ※1 A=ASCII B=バイナリ

        # read the t3w file
        with open(t3w_file, "rb") as f:
            t3w_bin_data = f.read()

            t3w_header_bin = t3w_bin_data[:1024]
            t3w_win32_bin = t3w_bin_data[1024:]

            # unpack the header
            # read only necessary part of the header
            # TODO: read all the header
            
            t3w_header_dict = {
                "device_program_name": struct.unpack(">12s", t3w_header_bin[4:16])[0].decode("utf-8"),
                "device_number": struct.unpack(">H", t3w_header_bin[24:26])[0],
                "num_channel": struct.unpack(">H", t3w_header_bin[30:32])[0],
                "sampling_time_interval": struct.unpack(">H", t3w_header_bin[40:42])[0],
                "delay_time": struct.unpack(">H", t3w_header_bin[42:44])[0],
                "sequence_number": struct.unpack(">H", t3w_header_bin[50:52])[0],
                "start_datetime_this_file": str(struct.unpack(">H", t3w_header_bin[52:54])[0]).zfill(2) + str(struct.unpack(">H", t3w_header_bin[54:56])[0]).zfill(2) \
                                            + str(struct.unpack(">H", t3w_header_bin[56:58])[0]).zfill(2) + str(struct.unpack(">H", t3w_header_bin[58:60])[0]).zfill(2) \
                                            + str(struct.unpack(">H", t3w_header_bin[60:62])[0]).zfill(2) + str(struct.unpack(">H", t3w_header_bin[62:64])[0]).zfill(2) \
                                            + str(struct.unpack(">H", t3w_header_bin[64:66])[0]).zfill(2),
                "start_datetime_first_file": str(struct.unpack(">H", t3w_header_bin[66:68])[0]).zfill(2) + str(struct.unpack(">H", t3w_header_bin[68:70])[0]).zfill(2) \
                                            + str(struct.unpack(">H", t3w_header_bin[70:72])[0]).zfill(2) + str(struct.unpack(">H", t3w_header_bin[72:74])[0]).zfill(2) \
                                            + str(struct.unpack(">H", t3w_header_bin[74:76])[0]).zfill(2) + str(struct.unpack(">H", t3w_header_bin[76:78])[0]).zfill(2) \
                                            + str(struct.unpack(">H", t3w_header_bin[78:80])[0]).zfill(2),
                "channel_offset_1": struct.unpack(">I", t3w_header_bin[224:228])[0],
                "channel_offset_2": struct.unpack(">I", t3w_header_bin[228:232])[0],
                "channel_offset_3": struct.unpack(">I", t3w_header_bin[232:236])[0],
                "latitude" : struct.unpack(">I", t3w_header_bin[808:812])[0] + struct.unpack(">I", t3w_header_bin[812:816])[0]/60,
                "longitude" : struct.unpack(">I", t3w_header_bin[816:820])[0] + struct.unpack(">I", t3w_header_bin[820:824])[0]/60,
                "north_south_flag": struct.unpack(">c", t3w_header_bin[828:829]),
                "east_west_flag": struct.unpack(">c", t3w_header_bin[829:830])
            }
            print(t3w_header_dict)
            t3w_win32_bin_info = {
                "start_datetime_frame" : struct.unpack(">8s", t3w_win32_bin[4:12])[0].hex(),
                "frame_duration" : struct.unpack(">I", t3w_win32_bin[12:16])[0],
                "data_block_size" : struct.unpack(">I", t3w_win32_bin[16:20])[0],
            }
            
            t3w_win32_bin_data = {}
            
            t3w_win32_bin_data["sample_size"] = int(bin(t3w_win32_bin[24]>>4), 0)
            t3w_win32_bin_data["data_size_in_channel_block"] = struct.unpack(">H", t3w_win32_bin[24:26])[0] & 0x0FFF
            
            # gain
            temp_gain = 2.048 / 2 ** 23
            
            # 32bit float data 
            t3w_win32_bin_data["initial_data"] = struct.unpack(">i", t3w_win32_bin[26:30])[0]
            temp_prev = t3w_win32_bin_data["initial_data"]
            print(temp_prev * temp_gain)
            
            for i in range(t3w_win32_bin_data["data_size_in_channel_block"]):
                
                temp_diff = struct.unpack(">h", t3w_win32_bin[30 + i*2:30 + i*2 + 2])[0]
                temp_cur_value = temp_prev + temp_diff
                temp_prev = temp_cur_value
                print(temp_cur_value * temp_gain)
            
            
            print(t3w_win32_bin_data)