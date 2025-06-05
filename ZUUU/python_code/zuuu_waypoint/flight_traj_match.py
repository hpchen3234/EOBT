import gc
import numpy  as np
import pandas as pd
import ast
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from geopy.distance import geodesic
from shapely.geometry import LineString,Point
from pyproj import Proj, Transformer
from itertools import chain
import networkx as nx
from shapely.wkt import loads
from rtree import index
#定义轨迹匹配类
class Rtree_match():
    # 初始数据  机场节点数据   航班轨迹数据
    def __init__(self,node_data,traj_data,bound ):
        self.node_data =node_data
        self.traj_data =traj_data
        self.traj_idx = None  #
        self.bound=bound
    def rtree_building(self):
        # 创建索引
        traj_idx=index.Index()
#         添加对象进Rtree
        for i,row in self.traj_data.iterrows():
            point=Point(row.xAxis,row.yAxis)
            traj_idx.insert(i,point.bounds)
        self.traj_idx=traj_idx

        return traj_idx


    def find_nearst(self,hist,node_point):
    #     hist是存储交叉点索引的集合
        id = -1
        a =  float('inf')
        for h in hist:
            traj_point = Point(self.traj_data.loc[h]['xAxis'], self.traj_data.loc[h]['yAxis'])
            # 轨迹点与机场点的距离
            dis = node_point.distance( traj_point)
            # print(dis)
            if dis < a:
                a = dis
                id = h
        return id

    def idx_match(self):
        matches=[]
        for _,node in self.node_data.iterrows():
            node_point=Point(node.xAxis,node.yAxis)
            # 节点外接（20，20的）缓冲区？
            buffered_area=node_point.buffer(self.bound)
            # 轨迹点与机场点缓冲区有交点：
            if len(list(self.traj_idx.intersection(buffered_area.bounds))) != 0:
                # print('1', node)
                # print('2',list(self.traj_idx.intersection(buffered_area.bounds)))
                # 交点集合
                hist = self.traj_idx.intersection(buffered_area.bounds)
                nearst_idx=self.find_nearst(hist,node_point)
                # 获取 self.traj_data 中对应行的完整信息
                traj_info = self.traj_data.loc[nearst_idx].to_dict()
                matches.append({
                    'trajectory_idx':nearst_idx,
                    'node_name': node['point'],
                    **traj_info
                })

                # 将匹配结果转换为 DataFrame
        matches_df = pd.DataFrame(matches)

        return matches_df

def construct_graph(points,gdf):
    G = nx.Graph()
    for index, row in points.iterrows():
        point_name = row['point']
        point_coords = (row['lon'], row['lat'])
        G.add_node(point_name, coords=point_coords)

    for idx, row in gdf.iterrows():
        roucode = row['roucode']  # 路段名称
        pointlist = row['pointlist']  # 节点名称列表
        geometry = row['geometry']  # LineString 几何对象

        # 获取 LineString 的坐标点
        coords = list(geometry.coords)
        # 确保 pointlist 和 coords 的长度一致
        if len(pointlist) != len(coords):
            raise ValueError(f"路段 {roucode} 的 pointlist 和 geometry 长度不匹配")
        # 添加节点和边
        for i in range(len(pointlist) - 1):
            start_node = pointlist[i]  # 起点名称
            end_node = pointlist[i + 1]  # 终点名称
            start_coord = coords[i]  # 起点坐标
            end_coord = coords[i + 1]  # 终点坐标
            # 添加节点（以节点名称为键，坐标为属性）
            if start_node not in G:
                G.add_node(start_node, pos=start_coord)
            if end_node not in G:
                G.add_node(end_node, pos=end_coord)
            # 添加边（起点到终点）
            G.add_edge(start_node, end_node, roucode=roucode)
    #
    # # 保存为 GraphML 文件
    # nx.write_graphml(G, r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\route_network.graphml'

    return G

def calculate_distance(row):
    point1 = (row['latitude'], row['longitude'])
    point2 = (row['latitude_2'], row['longitude_2'])
    return geodesic(point1, point2).meters

# matches=waypoint_matches.copy()
# graph=sta_route_graph
# matches=sid_matches.copy()
# graph=sid_star_graph
def check_route(matches,graph):
    # 添加一列 'connected'，初始值为 True
    matches['connected'] = True
    all_connected = True
    for i in range(len(matches) - 1):
        cur_node = matches['node_name'].iloc[i]
        nxt_node = matches['node_name'].iloc[i + 1]
        if nxt_node not in graph.neighbors(cur_node):
            # print(f"❌ 节点 {cur_node} 和 {nxt_node} 之间没有直接连接")
            matches.at[i, 'connected'] = False
            all_connected = False
    return  all_connected


def traj_filter(match_result,graph):
    matches=match_result.copy()
    # 记录需要删除的索引
    matches.reset_index(drop=True, inplace=True)  # 初始重置索引
    to_delete_indices = []
    for i, row in matches.iterrows():
        cur_node = row['node_name']
        neighbors = list(graph.neighbors(cur_node))
        # 1.首先删掉 两端都没有链接关系的匹配点
        if not any(neighbor in matches['node_name'].values for neighbor in neighbors):
            # print(f"节点 {cur_node} 的所有邻居节点都不在匹配结果中，删除该节点。")
            to_delete_indices.append(i)
    # 删除不满足条件的行
    matches.drop(to_delete_indices, inplace=True)
    # # 2.如果只有一段有连接关系
    # #   上一个点 与下一个有连接关系  那么删除该点
    # matches.reset_index(drop=True,inplace=True)  # 初始重置索引
    # to_delete_indices = []
    # for i, row in matches.iterrows():
    #     if i == 0 or i == len(matches) - 1:
    #         continue  # 跳过第一行和最后一行
    #     cur_node = row['node_name']
    #     up_node=matches['node_name'].iloc[i-1]
    #     nxt_node = matches['node_name'].iloc[i+1]
    #     print(up_node,cur_node, nxt_node)
    #     if  nxt_node  in list(graph.neighbors(up_node)):
    #
    #         print(f"节点 {up_node} 链接：{nxt_node}")
    #         to_delete_indices.append(i)
    #         print(f"节点 {cur_node} 多余，删除该节点。")
    # matches.drop(to_delete_indices, inplace=True)

    # 2：处理两点之间存在多个额外匹配的点
    matches.reset_index(drop=True,inplace=True)  # 初始重置索引
    i = 1  # 从索引 1 开始，因为需要访问上一个节点
    while i < len(matches) - 1:
        cur_node = matches['node_name'].iloc[i]
        up_node = matches['node_name'].iloc[i - 1]
        nxt_node = matches['node_name'].iloc[i + 1]
        # print(f"当前索引：{i}，当前节点 {cur_node}，上一个节点 {up_node}，下一个节点 {nxt_node}")
        # 检查当前节点与上一个节点是否有直接连接
        up_connected = up_node in list(graph.neighbors(cur_node))
        # print(f"节点 {cur_node} 和节点 {up_node} 之间是否有直接连接：{up_connected}")
        # 检查当前节点与下一个节点是否有直接连接
        nxt_connected = nxt_node in list(graph.neighbors(cur_node))
        # print(f"节点 {cur_node} 和节点 {nxt_node} 之间是否有直接连接：{nxt_connected}")

        # 如果当前节点与上一个节点或下一个节点没有直接连接
        if not up_connected or not nxt_connected:
            # print(f"节点 {cur_node} 与上一个节点或下一个节点没有直接连接，检查后续节点。")

            # 找到不连接的起始点
            start_index = i
            end_index = i + 1

            # 向后遍历，找到第一个与起始点有直接连接的节点

            while end_index < len(matches) :
                current_end_node = matches['node_name'].iloc[end_index]
                # 检查 up_node 和 current_end_node 是否有直接连接
                if up_node in list(graph.neighbors(current_end_node)):
                    # print(f"节点 {up_node} 和节点 {current_end_node} 之间有直接连接。")
                    to_delete_indices = list(range(start_index , end_index))
                    matches.drop(matches.index[to_delete_indices], inplace=True)
                    matches.reset_index(drop=True, inplace=True)
                    # print(f"删除无效节点后的匹配点：\n{matches}")
                    i = start_index   # 从删除的位置重新检查 位置不变
                    break  # 找到连接点，退出循环
                # print(f"节点 {up_node} 和节点 {current_end_node} 之间没有直接连接，检查点 {cur_node} 和节点 {current_end_node} 共同邻接点...")
                # # 获取 cur_node 和 current_end_node 的邻接点
                # cur_neighbors = set(graph.neighbors( cur_node))
                # end_neighbors = set(graph.neighbors(current_end_node))
                # # 查找共同邻接点
                # common_neighbors = cur_neighbors & end_neighbors
                # if common_neighbors:
                #     # 找出不存在于当前路径的共同邻接点
                #     available_nodes = [n for n in common_neighbors if n not in matches['node_name'].values]
                #
                #     if available_nodes:
                #         missing_node = available_nodes[0]  # 取第一个可用节点
                #         print(missing_node)
                #         # 在 cur_node 之后插入缺失点
                #         cur_pos = matches.index[matches['node_name'] == cur_node][0]
                #
                #         print(  cur_pos )
                #         # 创建新行数据（确保包含所有列）
                #         new_row = {col: None for col in matches.columns}  # 初始化所有列为None
                #         new_row['node_name'] = missing_node  # 设置要插入的节点名
                #         # 在指定位置插入
                #         # 正确插入新行（使用concat确保是插入而非替换）
                #         matches = pd.concat([
                #             matches.iloc[:cur_pos + 1],  # 当前节点及之前的所有行
                #             pd.DataFrame([new_row]),  # 新插入的行
                #             matches.iloc[cur_pos + 1:]  # 当前节点之后的所有行
                #         ], ignore_index=True)
                #
                #         print(f"补充后的匹配点：\n{matches}")
                #         i = start_index + 1  # 从添加点的位置重新检查 所以加一
                #         break  # 补充后重新检查路径
                else:
                    # print(f"无共同邻接点，继续向后检查...")
                    end_index+=1
            if end_index == len(matches):
                # print(f"节点 {up_node}与后续节点都无链接关系，")
                i+= 1  # 继续检查下一个节点
                        # 如果找到连接点，删除中间无效节点

        else:
            # print(f"节点 {cur_node} 与上一个节点和下一个节点都有直接连接，跳过。")
            i += 1

    # # 3.最后检查点与点之间是不是都是链接关系   证明路径正确

    all_connected=check_route(matches,graph)

    if all_connected:
        print("✅ 匹配路径正确，所有节点均直接连接")
        print("最终路径：", matches['node_name'].tolist())
    else:
        print("轨迹偏移较大，或没按程序飞行")
        print("最终路径：", matches['node_name'].tolist())
    return matches
        
def save_tocsv(file_path,data):
    if os.path.exists(file_path):
        data.to_csv(file_path, index=False, mode='a', header=False)
    else:
                    # 如果文件不存在，以写入模式写入数据，并包含表头
        data.to_csv(file_path, index=False)
        


if __name__=='__main__':
    # 读取处理后的轨迹 路网等数据
    # 1 整合一个月综合航迹数据  0126-0225===================================

    # folder_path= r"D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\SL_DCP1_sdi_all"  # 替换为你的日志文件夹路径


  
    # for filename in os.listdir(folder_path):
    #     if filename.endswith('.csv'):
    #         file_path = os.path.join(folder_path, filename)
    #         print(file_path)
    #         temp_df = pd.read_csv(file_path)  # 读取单个文件为 DataFrame
    #         traj_data= temp_df[( temp_df['ARR'] == 'ZUUU') | (temp_df['DEP'] == 'ZUUU')]
    #         traj_data.sort_values(by=['trackId', 'time'], inplace=True)
    #         # 删除重复行，确保 trackId 和 timestamp 的组合是唯一的
    #         traj_data.drop_duplicates(subset=['trackId', 'time', 'longitude', 'latitude'], keep='first', inplace=True)
    #         print('Df drop duplictate done,',  traj_data.shape)
    #         # traj_data = pd.concat([traj_data, temp_df], ignore_index=True)  # 合并到总 DataFrame 中    # 筛选出zuuu进离场航班
    #         # 保存进文件
    #         # 如果文件已存在，以追加模式写入数据
    #         csv_file_path = r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\zuuu_traj_0126-0225.csv'
    #         if os.path.exists(csv_file_path):
    #             traj_data.to_csv(csv_file_path, index=False, mode='a', header=False)
    #         else:
    #             # 如果文件不存在，以写入模式写入数据，并包含表头
    #             traj_data.to_csv(csv_file_path, index=False)
    #         print(f'Data written to CSV file: {csv_file_path}')
    #         del temp_df
    #         gc.collect()
  

    # zuuu_traj = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\zuuu_traj_0126-0225.csv', low_memory=False)
    # 太大了  读取一小部分
    zuuu_traj = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\zuuu_traj_0126-0225.csv', nrows=2000, low_memory=False)
    
    # zuuu_traj = traj_data[(traj_data['ARR'] == 'ZUUU') | (traj_data['DEP'] == 'ZUUU')]
    # 将时间统一变为时间戳形式
    # 将 'time' 列转换为 datetime 类型
    zuuu_traj['time'] = pd.to_datetime(zuuu_traj['time'])
    zuuu_traj['date'] =  zuuu_traj['time'].dt.date
    wgs84 = Proj(init='epsg:4326')  # WGS84 坐标系
    utm48n = Proj(init='epsg:32648')  # UTM 48N 坐标系
    # 使用 pyproj.Transformer 进行坐标转换
    transformer = Transformer.from_proj(wgs84, utm48n, always_xy=True)
    zuuu_traj[['xAxis', 'yAxis']] =zuuu_traj.apply(
        lambda row: transformer.transform(row['longitude'], row['latitude']),
        axis=1,
        result_type='expand'
    )
    zuuu_traj.sort_values(by=['time','callSign'],inplace=True)

    # 2.航路点数据  航路网
    waypoint_data=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\waypoint.csv')
    sta_route = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\waypoints_route.csv')

    # 3.STDSTAR数据 路网
    sid_star_point= pd.read_csv(fr'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\sid_star_point.csv')
    sid_star_info= pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\ZUUU_sid_star_route.csv')

    # 建图
    sta_route['pointlist'] = sta_route['pointlist'].apply(ast.literal_eval)
    sta_route['geometry'] = sta_route['geometry'].apply(loads)
    sta_route_graph = construct_graph(waypoint_data,sta_route)
    sid_star_info['pointlist'] = sid_star_info['pointlist'].apply(ast.literal_eval)
    sid_star_info['geometry'] = sid_star_info['geometry'].apply(loads)
    sid_star_graph = construct_graph(sid_star_point,sid_star_info)

    # 4.读取飞行计划
    fpl_data = pd.read_excel(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250225_tab_fpl_bak.xlsx')
    # 查看你plantype有哪些种类 ['OVERFLY' 'ARR' 'DRAG' 'TIP' 'DEP' 'TRACK' 'TAG']
    unique_platype = fpl_data['plantype'].unique()
    # 只保留 ARR DEP 类型
    zuuu_fpl = fpl_data[fpl_data['plantype'].isin(['ARR', 'DEP'])]
    
    # 删去atd  ata 是空值的行
    zuuu_fpl.dropna(subset=['atd','ata'],how='all',inplace=True)
    zuuu_fpl['route'] = zuuu_fpl['route'].str.replace(r'\bDCT\b', '', regex=True).str.strip().str.replace(' +', ' ',
                                                                                                          regex=True)
    
    
    
    zuuu_fpl['date']=  zuuu_fpl['atd'].dt.date
    # 将 route 列中的字符串分割为列表
    zuuu_fpl['route'] = zuuu_fpl['route'].str.split()
    # 对于每一行的route
    for idx, row in zuuu_fpl.iterrows():
        route = row['route']
        # 检查route是否为NaN
        if route is None or (isinstance(route, float) and np.isnan(route)):
            continue  # 如果是NaN，则跳过当前迭代
        # 修改异常值
        for i, info in enumerate(route):
            if '/' in info:
                route[i] = info.split('/')[0]
        zuuu_fpl.at[idx, 'route'] = route
    # zuuufpl_20250225 = zuuu_fpl[zuuu_fpl['atd'].dt.date == pd.Timestamp('2025-02-25').date()]
    zuuu_fpl.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\zuuu_fpl_0225.csv')


    # 轨迹匹配 == == == == == == == == == == == == == == == == ==
    # 初始化 all_match 为一个空的 DataFrame
    # all_match = pd.DataFrame()
    csv_file_path= r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\correct_match_result.csv'
    csv_file_path_1= r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\all_match_result.csv'
    csv_file_path_2= r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\unmatche_result.csv'
    route_match = []
    grouped=zuuu_traj.groupby(by=['date','callSign'])
    for (date,callSign),track_data in grouped:
        print(date,callSign)
        # 初始化 sid_matches 和 waypoint_matches 为 DataFrame
        sid_matches = pd.DataFrame()
        waypoint_matches = pd.DataFrame()

        sid_matches_copy = pd.DataFrame()
        waypoint_matches_copy = pd.DataFrame()

        # # # 一条示例
        # #
        # 
        # date,callSign = '2025-01-26','CCA2990'
        # track_data = zuuu_traj[zuuu_traj['date'] == date]
        # track_data = zuuu_traj[zuuu_traj['callSign'] == callSign]
        # track_data.to_csv(fr'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\track_data\{callSign}.csv', index=False)

        
        # 创建 Rtree_match 对象
        track_data.reset_index(inplace=True,drop=True)
        # 这里分两次 匹  一次匹进离场区域   另一次匹航路
        sid_matcher = Rtree_match(sid_star_point, track_data, 1200)
        # 构建轨迹点的 Rtree 索引
        sid_matcher.rtree_building()
        # 匹配机场节点和轨迹点z
        sid_matches = sid_matcher.idx_match()
        if not sid_matches.empty:
            sid_matches = sid_matches.sort_values(by='trajectory_idx')
            sid_matches.reset_index(inplace=True,drop=True)
            # 进一步处理 目前设的范围  存在有的点没匹配上 以及匹配多了的情况
            sid_matches_copy = sid_matches.copy()
            if check_route( sid_matches_copy, sid_star_graph):
                print("✅ 匹配路径正确，所有节点均直接连接")
                print("最终路径：", sid_matches_copy['node_name'].tolist())
            else:
                sid_matches_copy = traj_filter(sid_matches_copy, sid_star_graph)

            # 这里知道前面匹配到的轨迹段  由trajectory_idx表示 
            idx_part = [sid_matches['trajectory_idx'].iloc[0], sid_matches['trajectory_idx'].iloc[-1]]
            mask = (track_data.index <idx_part[0]) | (track_data.index >idx_part[1])
            # 过滤原始轨迹数据，移除指定范围内的轨迹点
            filter_track = track_data[mask]
        else:
            filter_track = track_data
        # 将轨迹筛掉前面进离场匹配的部分 在匹配航路点部分
        waypoint_matcher = Rtree_match(waypoint_data, filter_track, 5000)
        # 构建轨迹点的 Rtree 索引
        waypoint_matcher.rtree_building()
        # 匹配机场节点和轨迹点z
        waypoint_matches = waypoint_matcher.idx_match()
        if not waypoint_matches.empty:
            waypoint_matches = waypoint_matches.sort_values(by='trajectory_idx')
            waypoint_matches.reset_index(inplace=True,drop=True)
            waypoint_matches_copy=waypoint_matches.copy()
            if check_route(waypoint_matches_copy,sta_route_graph):
                    print("✅ 匹配路径正确，所有节点均直接连接")
                    print("最终路径：", waypoint_matches_copy['node_name'].tolist())
            else:
                waypoint_matches_copy = traj_filter(waypoint_matches, sta_route_graph)
        

        if not sid_matches_copy.empty and not waypoint_matches_copy.empty:
            merged_df = pd.concat([sid_matches_copy, waypoint_matches_copy], ignore_index=True)
        elif not sid_matches_copy.empty:
            merged_df = sid_matches_copy
        elif not waypoint_matches_copy.empty:
            merged_df = waypoint_matches_copy
        else:
            continue

            # 如果合并后的数据只有一行，跳过当前迭代
        if len(merged_df) <= 1:
            continue
        # 删除重复行，优先保留 sid_matches_copy 中的行
     
        merged_df = merged_df.drop_duplicates(subset=['node_name'], keep='first')
        # 然后根轨迹索引排序
        merged_df= merged_df.sort_values(by='trajectory_idx')

        # 记录在 route——match里面  并且随着循环增加 记录 呼号  日期 路径

        match_route= merged_df ['node_name'].tolist()
        route_match.append({
            'callSign': callSign,
            'date': date,
            'match_route': match_route
        })
        # 创建一个新的 DataFrame，将每一行与其下一行拼接
        shifted_df = merged_df.shift(-1)  # 将 df 向上移动一行，最后一行会变成 NaN
        shifted_df.columns = [f"{col}_2" for col in shifted_df.columns]  # 修改列名，加上 _2
        # 将原始 DataFrame 和移动后的 DataFrame 拼接
        result_df = pd.concat([merged_df, shifted_df], axis=1)
        # 删除最后一行，因为最后一行的下一行是 NaN
        result_df = result_df[:-1]
        original_columns = merged_df.columns.tolist()
        # 创建新的列顺序
        new_columns = []
        for col in original_columns:
            new_columns.append(col)
            new_columns.append(f"{col}_2")
        # 重新排列列的顺序
        result_df = result_df[new_columns]


        # 两段相隔的距离  飞行时间
        # 应用函数计算距离并添加到 DataFrame 中
        result_df['distance'] = result_df.apply(calculate_distance, axis=1)
        result_df['time'] = pd.to_datetime(result_df['time'])
        result_df['time_2'] = pd.to_datetime(result_df['time_2'])

        # 计算时间差
        result_df['time_delta'] =( result_df['time_2'] - result_df['time']).dt.total_seconds()

        result_df['mean_speed'] =result_df['distance'] / result_df['time_delta']

        result_df = result_df.drop(columns=['date_2','trackId','trackId_2','callSign_2','code','code_2','warnFlag_2','ARR_2','DEP_2','EOBT_2','RVSM_2','FR_2','acftTurb_2','timestamp', 'timestamp_2', 'xAxis',
       'xAxis_2', 'yAxis', 'yAxis_2','connected_2'])
        # 还需要添加  计划里对应的信息
        # 选择需要的列
        columns_to_merge = ['date','arcid', 'plantype', 'arctyp', 'wktrc', 'cruisespd', 'cruisehit']

        # 合并数据，指定匹配的列
        result_df = result_df.merge(zuuu_fpl[columns_to_merge], left_on=['date','callSign'], right_on=['date','arcid'], how='left')

        # 删除多余的 arcid 列
        result_df = result_df.drop(columns=['arcid'])
        # 把date callsign两列放进最前面
      
        result_df = result_df[['date', 'callSign','connected'] + [col for col in result_df.columns if col not in ['date', 'callSign','connected']]]
        save_tocsv(csv_file_path_1,result_df)
        if check_route(waypoint_matches_copy,sta_route_graph)and check_route( sid_matches_copy, sid_star_graph):
            correct_df= result_df
            save_tocsv(csv_file_path,correct_df)
        else:
            unmatch_df= result_df
            save_tocsv(csv_file_path_2,unmatch_df)
        # # 得到一个写进一个 将文件写进CSV

    
       
       
       
        # all_match = pd.concat([all_match, result_df], ignore_index=True)

    # route_match.to