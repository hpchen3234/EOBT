
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import LineString
from ZUTF.code.zutf_match import Rtree_match
from ZUUU.python_code.zuuu_airport import geopoint_trans,is_straight_or_curve
from ZUUU.python_code.flightplan_route import build_matrix
# 定义转换函数
def subtract_local_utm(geom, local_utm):
    if geom.geom_type == 'LineString':
        # 获取LineString的所有坐标
        coords = list(geom.coords)
        # 减去local_utm的坐标
        new_coords = [(x - local_utm[0], y - local_utm[1]) for x, y in coords]
        # 创建新的LineString对象
        return LineString(new_coords)
    else:
        return geom  # 如果不是LineString，返回原几何对象

def delete_point(flght):
    #
    # 设置时间间隔阈值（一秒为间隔）
    time_gap_threshold = pd.Timedelta(seconds=5)

    # 计算相邻点之间的时间间隔
    flght['time_diff'] = flght['timestamp'].diff()

    # 标记时间间隔超过阈值的点
    flght['is_discontinuous'] = flght['time_diff'] > time_gap_threshold
    flght['segment_id'] = (  flght['is_discontinuous'].cumsum()).astype(int)
    return flght

    # # 找到不连续的轨迹段
    # discontinuous_points = flght[flght['is_discontinuous']].index.tolist()
    # # 分割轨迹
    # trajectory_segments = []
    # start_index = flght.index[0]
    # for end_index in discontinuous_points:
    #     if end_index==flght.index[-1]:
    #         break
    #     else:
    #         segment = flght.loc[start_index:end_index-1]
    #         if not segment.empty:
    #             trajectory_segments.append(segment)
    #         start_index = end_index
    #
    # # 添加最后一段轨迹
    # last_segment = flght.loc[start_index:]
    # if not last_segment.empty:
    #    trajectory_segments.append(last_segment)
    #
    #
    # return  trajectory_segments


def plot_traj(id):
        flght = traj_data[traj_data['trackId'] == id]

        # transformer = Transformer.from_crs("epsg:4326", "epsg:32648", always_xy=True)
        # # 一次性转换所有坐标
        # flght['xprj'], flght['yprj'] = transformer.transform(flght['longitude'].values, flght['latitude'].values)

        # 轨迹与机场底图
        fig, ax = plt.subplots(figsize=(10, 10))

        airport_graph['geometry'].plot(ax=ax, color='lightblue', linewidth=2, label="Transformed", alpha=0.7)
        plt.scatter(flght['xAxis'], flght['yAxis'], label='traj Coordinates')

        # plt.scatter(flght['xAxis'], flght['yAxis'], label='XY Coordinates')
        # plt.scatter(flght_2['xAxis'], flght_2['yAxis'], label='XY Coordinates')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.legend()
        plt.show(block=True)

if __name__=='__main__':

    # 好像不需要转换坐标系 机场底图 与轨迹是同一个坐标系
    # 直接使用xml文件中的中心点  以下是度分秒格式   转为小数点格式
    #         <node tip="经度" param="LONGITUDE" value="E10357020"/>
    #         <node tip="纬度" param="LATITUDE" value="N3034470"/>
    loacl_lon, local_lat=(103.95055555555555, 30.579722222222223)
    # 转为UTM投影
    local_utm=(399374.24775233655, 3383494.9694276787, 48, 'R')
  
    airport_node = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_shp\zuuu_node_del.csv')

    airport_node['xAxis']=airport_node['x']-local_utm[0]
    airport_node['yAxis'] = airport_node['y'] - local_utm[1]
#     机场底图
    airport_graph= gpd.read_file(r'D:\eobt_Project\ZUUU\zuuu_shp\graphics.shp')
#     把机场底图对应的坐标信息也统一 变成中心投影
    airport_graph['geometry'] = airport_graph['geometry'].apply(lambda geom: subtract_local_utm(geom, local_utm))
    airport_graph=geopoint_trans(airport_graph)
    airport_graph['distance']=  airport_graph.geometry.apply(lambda x:x.length)
    airport_graph['road_shape'] = airport_graph.geometry.apply(is_straight_or_curve)
    airport_graph.to_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\zuuu_gragh.csv')


    # 读取轨迹数据
    # 这里需要写一个循环读取一个月的文件
    folder_path = r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\SL_MSDP2_SMRDP_AIRNET'
    files = os.listdir(folder_path)
    start_date = datetime(2025, 2, 3)
    end_date = datetime(2025, 3, 5)
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    # 遍历日期范围
    for date in tqdm(date_range, desc="Processing Dates", unit="date"):
        # 格式化日期为字符串
        target_date = date.strftime("%Y%m%d")
        print(f"Processing date: {target_date}")
        # 筛选出目标日期和时间段的日志文件
        target_file = f"smrdp_{target_date}.csv"
        file_path = os.path.join(folder_path, target_file )
        traj_data= pd.read_csv(file_path)
        # 简化列的名字
        traj_data.columns=[col.split('.')[-1] for col in traj_data.columns]
        traj_data.insert(0, 'timestamp', pd.to_datetime(traj_data['tvSec'], unit='s'))

        traj_data.dropna(subset='trackId', inplace=True)
        traj_data.dropna(subset=['xAxis', 'yAxis'], inplace=True)
        traj_data['trackId'] = traj_data['trackId'].astype(int)
        # # filtered_traj_data =  filtered_traj_data.reset_index(drop=True)
        # # traj_data.rename(columns={'xAxis': 'nx', 'yAxis': 'ny'}, inplace=True)
        # # 计算每个trackId和日期的组合出现次数
        # unique_track_count = traj_data['trackId'].nunique()
        # # # 统计每个 trackId 的轨迹点个数
        # track_traj_count = traj_data.groupby(['trackId']).size().reset_index(name='count')
        # # 筛选出现次数超过 200 的组合
        # valid_combinations = track_traj_count[track_traj_count['count'] > 200]['trackId']
        # # 保留 traj_data 中符合条件的记录
        # filtered_traj_data = traj_data.merge(valid_combinations, on='trackId')

        all_matches = []
        flight_group =traj_data.groupby(by='trackId')
        for trackid, group in tqdm(flight_group, desc="Processing tracks", unit="track"):
            # 删除 trackid 轨迹点过少的
            if len(group)<200:
                continue
            else:
                # 存在同一id 多个时间段的轨迹   进行轨迹分段
                group=delete_point(group)
                for segment_id, child_group in group.groupby('segment_id'):
                    #         删去时间重复点  以及轨迹重复点
                    child_group.drop_duplicates(subset='timestamp')
                    child_group.drop_duplicates(subset=['xAxis','yAxis'])
                    if len(child_group)<200:
                        continue
                    else:
                    #     进行匹配
                        # 把轨迹做r树
                        # 创建 Rtree_match 对象
                        matcher = Rtree_match(airport_node, child_group)
                        # 构建轨迹点的 Rtree 索引
                        matcher.rtree_building()
                        # 匹配机场节点和轨迹点z
                        matches = matcher.idx_match()
                        matches['trackid'] = trackid
                        matches['segment_id'] = segment_id

                        if not matches.empty:
                            matches = matches.sort_values(by='trajectory_idx')

                        all_matches.append(matches)


        # 将所有的匹配结果合并成一个完整的 DataFrame
        final_matches_df = pd.concat(all_matches, ignore_index=True)

        # final_matches_df.to_csv(r'D:\eobt_Project\ZUUU\traj_route.csv')

        # ==============================================这里还是需要添加一个轨迹路段的检查处理
        # 如果匹配出来的轨迹是不连续、跳跃的 具体看轨迹怎么回事   不对的删了
        # 如果是多匹配了几个点  就具体分析删去 多余点

       # 构建邻接矩阵和邻接字典
        adj_matrix, adj_dict = build_matrix(airport_graph)
        # 按航班和日期分组
        grouped = final_matches_df.groupby(by=['trackid', 'segment_id'])
        # 初始化一个空列表，用于存储每个过滤后的 DataFrame
        checked_route = []
        for (trackid, segment_id), group in grouped:
            copy_group = group.copy()  # 创建副本以避免直接修改原始数据
            # 初始化布尔索引数组，初始值为 True
            keep_indices = [True] * len(copy_group)
            # 第一步：删除中间的点 p2
            for i in range(len(copy_group) - 2):  # 遍历到倒数第三个点
                p1 = str(int(copy_group['airport_node'].iloc[i]))
                p2 = str(int(copy_group['airport_node'].iloc[i + 1]))
                p3 = str(int(copy_group['airport_node'].iloc[i + 2]))
                # 如果 p1 和 p3 有连接关系，标记中间的 p2 为删除
                if p3 in adj_dict[p1]:
                    keep_indices[i + 1] = False
            # 使用布尔索引过滤 DataFrame
            filtered_group = copy_group[keep_indices]
            # 第二步：检查过滤后的点之间是否有连接关系
            for i in range(len(filtered_group) - 1):
                p1 = str(int(filtered_group['airport_node'].iloc[i]))
                p2 = str(int(filtered_group['airport_node'].iloc[i + 1]))
                # 检查 p1 和 p2 是否有连接关系
                if p2 not in adj_matrix[p1]:
                    print(f"Warning: No connection between {p1} and {p2} in trackid={trackid}, segment_id={segment_id}")
                    # 可以选择进一步处理，例如删除 p2 或记录问题

            # 将过滤后的 DataFrame 添加到列表中
            checked_route.append(filtered_group)
        # 将所有过滤后的 DataFrame 合并为一个大的 DataFrame
        final_checked_route = pd.concat(checked_route, ignore_index=True)

        # 这里需要截取跑道上的点
        rwyin_line = airport_graph[airport_graph['type'] == 'rwyline']
        rwy_set = pd.concat([rwyin_line['point_1'], rwyin_line['point_2']]).unique()


        # 新建一个df
        # 每架航班  每个路段上的信息统计
        fp_seg_info = pd.DataFrame(
            columns=['trackid', 'segment_id', 'node_1', 'node_2',  'timestamp1','t1', 'timestamp2','t2','seg_distance', 'seg_type',
                     'mean_speed'])

        for (trackid, segment_id), group in  final_checked_route.groupby(by=['trackid', 'segment_id']):
             # 每两点进行比较计算

             for i in range(0,len(group)-1):

                 p1=str(int(group['airport_node'].iloc[i]))
                 p2=str(int(group['airport_node'].iloc[i+1]))

                 if  p1 in rwy_set and p2 in rwy_set:
                     continue
                 idx1=int(group['trajectory_idx'].iloc[i])
                 idx2 = int(group['trajectory_idx'].iloc[i+1])
    # #              将索引对应到轨迹上去

                 t1=traj_data.loc[idx1]['tvSec']
                 t2=traj_data.loc[idx2]['tvSec']
                 timestamp1=group['timestamp'].iloc[i]
                 timestamp2 = group['timestamp'].iloc[i+1]

                # 对应路段的长度  需要对应到机场路网数据上
                 # 查询机场路网数据中的路段长度，不考虑 point_1 和 point_2 的顺序
                 matching_rows = airport_graph[
                     (airport_graph['point_1'] == p1) & (airport_graph['point_2'] == p2) |
                     (airport_graph['point_1'] == p2) & (airport_graph['point_2'] == p1)
                     ]
                 if matching_rows.empty:
                     continue
                 seg_distance = matching_rows['distance'].values[0]
                 seg_type = matching_rows['road_shape'].values[0]

    #            求解速度的两种算法
    #              1 长度除以时间
                 mean_speed=seg_distance/(t2-t1)
    #
    #              把上述信息加入df中
                 # 将上述信息加入 DataFrame 中
                 fp_seg_info = fp_seg_info._append({
                     'trackid': trackid,
                     'segment_id':segment_id,
                     'node_1': p1,
                     'node_2': p2,
                     'timestamp1':timestamp1,
                     't1':t1,
                     'timestamp2': timestamp2,
                     't2':t2,
                     'seg_distance': seg_distance,
                     'seg_type': seg_type,
                     'mean_speed': mean_speed,
                 }, ignore_index=True)
        save_path = fr'D:\eobt_Project\ZUUU\traj_match\traj_{target_date}.csv'
        fp_seg_info.to_csv(save_path, index=False)
