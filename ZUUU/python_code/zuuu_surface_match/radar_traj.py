from ZUTF.code.trajectory_data import parse_log,read_xml,smr_xml,polar_to_geodetic
import numpy as np
import pandas as pd
import re
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import  os
from pyproj import Proj, transform,Transformer
import math
import geopandas as gpd
from  ZUTF.code.zutf_match import transform_linestring
import  utm




def duplicate_point(data_df):
    # 按照航班以及日期分类
    grouped =data_df.groupby(['trackId', data_df['timestamp'].dt.date])
    # 创建一个列表来存储需要保留的 trackId
    result_df = pd.DataFrame()
    for id, group in grouped:
        group = group.sort_values(by='timestamp')
        # 保留每个重复序列的第一个和最后一个点
        un_group = group.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
        un_group = pd.concat(
            [un_group, group.drop_duplicates(subset=['latitude', 'longitude'], keep='last')])
        # 最后再去掉一个重复的
        un_group=un_group.drop_duplicates(subset=['tvSec', 'trackId'],keep='first')
        un_group =  un_group.sort_values(by='timestamp')
        # 将处理后的数据添加到结果 DataFrame 中
        result_df = pd.concat([result_df, un_group], ignore_index=True)

    return result_df

   # 定义一个函数来转换经纬度到UTM，并只返回东坐标和北坐标
def latlon_to_utm(row):
        utm_result = utm.from_latlon(row['latitude'], row['longitude'])
        return pd.Series([utm_result[0], utm_result[1]])  # 返回东坐标和北坐标



if __name__=='__main__':
    # # =======================================================================打开日志文件
    #时间跨度 就四天
    log_dicts=[]
    # 指定日志文件夹路径
    log_folder = r'D:\eobt_Project\ZUUU NEW\zuuu_log'

    # 遍历文件夹中的所有文件
    for filename in os.listdir(log_folder):
        # 如果是 .log 文件
        if filename.endswith('.log'):
            file_path = os.path.join(log_folder, filename)
            # 读取并解析每个文件
            with open(file_path, 'r') as file:
                for line in file:
                    line_dict = parse_log(line)
                    if line_dict:
                        log_dicts.append(line_dict)



    # 转为df
    log_df = pd.DataFrame(log_dicts)
    log_df =log_df.sort_values(by=['trackId','timestamp'])
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
    # 统计一下到原点的距离
    log_df['distance'] = log_df.apply(lambda row: math.sqrt(row['xAxis'] ** 2 + row['yAxis'] ** 2), axis=1)
    # 读取xml 系统 雷达中心点文件
    loacl_lon,local_lat=read_xml(r'D:\eobt_Project\ZUUU NEW\zuuu_log\itwr_global_offline.xml')
    radar_df = smr_xml(r'D:\eobt_Project\ZUUU NEW\zuuu_log\itwr_sfp_smr_offline.xml')
    trajcode_0 = log_df[log_df['code'] == 0].copy()
    # 筛选出 code 为 -1 的记录
    trajcode_1 = log_df[log_df['code'] == -1].copy()  # 使用 .copy() 避免 SettingWithCopyWarning
    t_1 = trajcode_1.copy()
    t_0 = trajcode_0.copy()


    # ==============转换坐标 （1.根据中心点以及极坐标转换  结果发现轨迹点有误差
    ''' 
    trajcode_0['srcID'] = trajcode_0['srcID'].astype(int).astype(str)
    df_merged = trajcode_0.merge(radar_df[['INDEX', 'LONGITUDE', 'LATITUDE']], left_on='srcID',right_on='INDEX', how='left')
    # 转换坐标
    df_merged[['longitude','latitude']]=df_merged.apply(lambda row:
                                                        polar_to_geodetic( row['LATITUDE'],
                                                                           row['LONGITUDE'],
                                                                           row['rho'],
                                                                           row['theta']),axis=1,
                                                        result_type='expand')
    # 确保索引匹配
    df_merged = df_merged.reset_index(drop=True)
    trajcode_0 = trajcode_0.reset_index(drop=True) 
     trajcode_0[['longitude','latitude']]= df_merged[['longitude','latitude']]
    '''

    #===============.坐标转换2  根据中心点进行UTM投影  再平移转换所有轨迹点的投影坐标  最后转为经纬度
    # 经纬度 ---utm投影!!!
    # 中心点投影坐标 带号48 R EPSG:32648
    local_utm = utm.from_latlon(local_lat, loacl_lon)
    t_0['prj_x'] = t_0['xAxis'] + local_utm[0]
    t_0['prj_y'] = t_0['yAxis'] + local_utm[1]
    # 创建坐标转换器
    transformer = Transformer.from_crs(f"EPSG:32648", "EPSG:4326", always_xy=True)
    t_0['longitude'], t_0['latitude'] = transformer.transform(t_0['prj_x'], t_0['prj_y'])
    t_0['n_x'] = t_0['prj_x'] - local_utm[0]
    t_0 = t_0.reset_index(drop=True)
    trajcode_0 = trajcode_0.reset_index(drop=True)
    trajcode_0[['longitude', 'latitude']] = t_0[['longitude', 'latitude']]

    # 这是在code=-1上验证 发现转换方法是对的
    t_1[['prj_x', 'prj_y']] = t_1.apply(latlon_to_utm, axis=1)
    t_1['n_x'] = t_1['prj_x'] - local_utm[0]
    t_1['n_y'] = t_1['prj_y'] - local_utm[1]
    t_1['n_lon'], t_1['n_lat'] = transformer.transform(t_1['prj_x'], t_1['prj_y'])

    trajcode_0 .to_csv(r'D:\eobt_Project\ZUUU NEW\zuuu_log\zuuu_trajdata0_origin.csv')
    trajcode_1 .to_csv(r'D:\eobt_Project\ZUUU NEW\zuuu_log\zuuu_trajdata1_origin.csv')

    # ==================================删除重复点
    result_df_1 = duplicate_point(trajcode_1)
    result_df_0 = duplicate_point(trajcode_0)

    result_df_0.to_csv(r'D:\eobt_Project\ZUUU NEW\zuuu_log\zuuu_trajdata0.csv')
    result_df_1.to_csv(r'D:\eobt_Project\ZUUU NEW\zuuu_log\zuuu_trajdata1.csv')

    # ===========================================画图验证
    flght = result_df_1[result_df_1['trackId'] == 1313]
    flght_2 = trajcode_0[trajcode_0['trackId'] == 16]
    # flght_2 = t_0[t_0['trackId'] == 16]
    # 机场底图
    zuuu_geo = gpd.read_file(r'D:\eobt_Project\ZUUU NEW\zuuu_shp\zuuu_graphics_geo.shp')
    # 轨迹与机场底图
    fig, ax = plt.subplots(figsize=(10, 10))
    zuuu_geo['geometry'].plot(ax=ax, color='lightblue', linewidth=2, label="Transformed", alpha=0.7)
    plt.scatter(flght['longitude'], flght['latitude'], label='geo Coordinates')
    plt.scatter(flght_2['longitude'], flght_2['latitude'], label='geo Coordinates')
    # plt.scatter(flght['xAxis'], flght['yAxis'], label='XY Coordinates')
    # plt.scatter(flght_2['xAxis'], flght_2['yAxis'], label='XY Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()
    plt.show(block=True)






