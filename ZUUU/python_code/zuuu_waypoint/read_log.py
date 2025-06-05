import geopandas as gpd
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
import pandas as pd
import os
from tqdm import tqdm
import gc

def extract_and_organize_log(file_path):



    # 正则表达式模式
    pattern = re.compile(
        r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}) \[INFO \] RECV process: IN_FROM_TMA: "
        r"(?:trackId=(?P<trackId>\d+), )?"
        r"callSign=\[(?P<callSign>\w+)\], ?"
        r"(?:code=(?P<code>\d+), )?"
        r"(?:h0=(?P<h0>\d+), )?"
        r"(?:h=(?P<h>\d+), )?"
        r"(?:QNH Applied, )?"
        r"(?:CFL=(?P<CFL>\d+), )?"
        r"(?:longitude=(?P<longitude>[\d\.]+), )?"
        r"(?:latitude=(?P<latitude>[\d\.]+), )?"
        r"(?:SPEED=(?P<SPEED>\d+), )?"
        r"(?:HEADING=(?P<HEADING>\d+), )?"
        r"(?:warnFlag=(?P<warnFlag>\d+), )?"
        r"(?:ARR=(?P<ARR>\w+), )?"
        r"(?:DEP=(?P<DEP>\w+), )?"
        r"(?:EOBT=(?P<EOBT>\d+), )?"
        r"(?:RVSM=(?P<RVSM>\w+), )?"
        r"(?:FR=(?P<FR>\w+), )?"
        r"(?:acftTurb=(?P<acftTurb>\w+), )?"
        r"(?:timestamp=(?P<timestamp>.*))?"
    )

    # 用于存储提取的结构化数据
    extracted_data = []
    with open(file_path, "r") as file:
        log_content = file.readlines()
    for line in tqdm(log_content,desc="Processing lines", unit="line"):
        if "[INFO ] RECV process: IN_FROM_TMA:" in line:
            match = pattern.search(line)
            if match:
                # 提取所有可能的字段，如果字段缺失则设置为 None
                data = match.groupdict()

                extracted_data.append(data)
    print('extract done')

    # save_path = os.path.splitext(file_path)[0] + ".csv"
    #
    # # 保存为 CSV 文件
    # df.to_csv(save_path, index=False)
    return extracted_data


def process_logs_in_folder(folder_path, target_date):
    """
    批量处理文件夹中的日志文件，并提取指定日期和时间段的数据
    :param folder_path: 日志文件夹路径
    :param target_date: 目标日期（格式为 YYYYMMDD）
    :param start_hour: 开始小时（00-23）
    :param end_hour: 结束小时（00-23）
    :return: 合并后的 DataFrame
    """
    # 列出文件夹中的所有文件
    files = os.listdir(folder_path)

    # 筛选出目标日期和时间段的日志文件
    target_files = [
        f for f in files
        if f.startswith(f"SL_DCP1_sdi_{target_date}_") and f.endswith(".log")
    ]

    # # 进一步筛选出指定时间段的文件
    # target_files = [
    #     f for f in target_files
    #     if int(f.split("_")[-1].split(".")[0]) >= start_hour and
    #        int(f.split("_")[-1].split(".")[0]) <= end_hour
    # ]

    # 初始化一个空的 DataFrame
    combined_df = pd.DataFrame()

    # 遍历目标文件并提取数据

    for file_name in tqdm( target_files, desc="Processing files", unit="file_name"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")
        df = extract_and_organize_log(file_path)
        combined_df = pd.concat([combined_df, df])

    # 按照 trackId 和 timestamp 排序
    combined_df.sort_values(by=['trackId', 'time'], inplace=True)
    # 删除重复行，保留第一条记录
    combined_df.drop_duplicates(subset=['trackId',  'time', 'longitude', 'latitude'], keep='first', inplace=True)

    return combined_df

    #

def plot_trajectory(zuuu_geo,df, track_id):
        """
        结合机场
        绘制指定 trackId 的轨迹图
        :param df: 包含轨迹数据的 DataFrame
        :param track_id: 要绘制轨迹的 trackId
        """
        # 筛选出指定 trackId 的数据
        track_data = df[df['trackId'] == track_id]

        if track_data.empty:
            print(f"No data found for trackId {track_id}.")
            return

        # 提取经纬度数据
        longitudes = track_data['longitude'].astype(float)
        latitudes = track_data['latitude'].astype(float)
        timestamps = track_data['timestamp']

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制机场地理信息
        zuuu_geo.plot(ax=ax, color='lightblue', linewidth=2, label="Airport Geometry", alpha=0.7)

        # 绘制轨迹
        ax.plot(longitudes, latitudes, marker='o', linestyle='-', label=f'Track ID: {track_id}')
        ax.scatter(longitudes, latitudes, color='blue')  # 添加散点图表示轨迹点

        # # 添加时间戳注释
        # for i, timestamp in enumerate(timestamps):
        #     ax.text(longitudes.iloc[i], latitudes.iloc[i], f"{timestamp:%H:%M:%S}", fontsize=8)

        # 设置图表标题和标签
        ax.set_title(f'Trajectory of Track ID: {track_id}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.grid(True)


        plt.show(block=True)

if __name__=='__main__':
    # db_config = {
    #     'username': 'root',
    #     'password': '123456',
    #     'host': 'localhost',
    #     'port': '3306',
    #     'database': 'ZUUU_SDI'
    # }
    # # 创建数据库连接
    # engine = create_engine(f"mysql+mysqlconnector://{db_config['username']}:{db_config['password']}@"
    #                        f"{db_config['host']}:{db_config['port']}/{db_config['database']}")

    # 示例：批量处理日志文件并提取指定日期和时间段的数据
    # 日志文件夹路径
    log_folder =  r"D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\SL_DCP1_sdi_all"  # 替换为你的日志文件夹路径
    # target_date = "20250225"  # 替换为你想处理的日期
    for filename in os.listdir(log_folder):
        if filename.endswith('.log'):
            file_path = os.path.join(log_folder, filename)
            print(f'Processing file: {file_path}')
            date_part =  filename.split('_')[3].split('.')[0]  # 假设文件名格式为 SL_DCP1_sdi_YYYYMMDD_HH.log
            print(date_part)
            table_name = f"sl_dcp1_sdi_{date_part}"  # 表名以日期命名
            extracted_data= extract_and_organize_log(file_path)
            df = pd.DataFrame( extracted_data)
            print('Df done')
            # 将 timestamp 转换为 datetime 类型
            df['time'] = pd.to_datetime(df['time']).dt.floor('s')
            # # 按照 trackId 和 timestamp 排序
            df.sort_values(by=['trackId', 'time'], inplace=True)
            # 删除重复行，确保 trackId 和 timestamp 的组合是唯一的
            df.drop_duplicates(subset=['trackId', 'time', 'longitude', 'latitude'], keep='first', inplace=True)
            print('Df drop duplictate done,',df.shape)
            # 将 DataFrame 写入 CSV 文件
            csv_file_name = f"sl_dcp1_sdi_{date_part}.csv"  # CSV 文件名以日期命名
            csv_file_path = os.path.join(log_folder, csv_file_name)

            # 如果文件已存在，以追加模式写入数据
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, index=False, mode='a', header=False)
            else:
                # 如果文件不存在，以写入模式写入数据，并包含表头
                df.to_csv(csv_file_path, index=False)

            print(f'Data written to CSV file: {csv_file_path}')

            # 删除处理过的日志文件

            os.remove(file_path)
            print(f'Deleted file: {file_path}')
            del extracted_data
            del df
            gc.collect()
            # chunk_size = 10000
            # for i in range(0, len(extracted_data), chunk_size):
            #     chunk = extracted_data[i:i + chunk_size]
            #     df = pd.DataFrame(chunk)
            #     # 将 timestamp 转换为 datetime 类型
            #     df['time'] = pd.to_datetime(df['time']).dt.floor('s')
            #     # 按照 trackId 和 timestamp 排序
            #     df.sort_values(by=['trackId', 'time'], inplace=True)
            #     # 删除重复行，确保 trackId 和 timestamp 的组合是唯一的
            #     df.drop_duplicates(subset=['trackId', 'time', 'longitude', 'latitude'], keep='first', inplace=True)
            #     # # 将 DataFrame 写入数据库
            #     # df.to_sql(table_name, con=engine, index=False, if_exists='append', chunksize=1000)



    # start_hour = 0  # 开始小时
    # end_hour = 9 # 结束小时（提取从00:00到02:00的数据）
    #
    # df = process_logs_in_folder(folder_path, target_date)
    # path_save=os.path.join( folder_path,f'{target_date}.csv')
    # df.to_csv(path_save, index=False)

    # # 机场数据读取
    # airport_graph = gpd.read_file(r'D:\eobt_Project\ZUUU\zuuu_shp\zuuu_graphics_geo.shp')
    #
    # # 示例：绘制特定 trackId 的轨迹图
    # track_id = '1082'  # 替换为你想绘制的 trackId
    # track_data1 = df[df['trackId'] == track_id]
    # plot_trajectory(airport_graph,df, track_id)




