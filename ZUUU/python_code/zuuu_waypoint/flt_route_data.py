import geopandas as gpd
import re
import pandas as pd
import ast
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# import plotly.graph_objects as go
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString,Point
from pyproj import Proj, Transformer
from itertools import chain
from ZUTF.code.zutf_match import Rtree_match
import networkx as nx

from shapely.wkt import loads

def coor_convert(coorstr):
    direction = coorstr[0]
    num_part =coorstr[1:]
    if direction=='E':
        # 判断格式并解析
        if '.' in num_part:
            # 情况1：带小数点的格式（可能是 E1034326.6 或 E10354.5）
            parts = num_part.split('.')
            integer_part = parts[0]
            decimal_part = parts[1]

            if len(integer_part)>5:
                # 格式 E1034326.6 → 103度43分26.6秒
                degrees = float(integer_part[:3])
                minutes = float(integer_part[3:5])
                seconds = float(integer_part[5:] + '.' + decimal_part)
            else:
                # 格式 E10354.5 → 103度54.5分
                degrees = float(integer_part[:3])
                minutes = float(integer_part[3:] + '.' + decimal_part)
                seconds = 0.0
        else:
            # 格式 E1051218 → 105度12分18秒
            degrees = float(num_part[:3])
            minutes = float(num_part[3:5])
            seconds = float(num_part[5:])

        # 计算十进制经度
        decimal = degrees + minutes / 60 + seconds / 3600

    # 处理方向（东经为正，西经为负）
    if direction == 'N':
        # 判断格式并解析
        if '.' in num_part:
            # 情况1：带小数点的格式（可能是 E1034326.6 或 E10354.5）
            parts = num_part.split('.')
            integer_part = parts[0]
            decimal_part = parts[1]

            if len(integer_part) > 4:

                degrees = float(integer_part[:2])
                minutes = float(integer_part[2:4])
                seconds = float(integer_part[4:] + '.' + decimal_part)
            else:
                # 格式 E10354.5 → 103度54.5分
                degrees = float(integer_part[:2])
                minutes = float(integer_part[2:] + '.' + decimal_part)
                seconds = 0.0
        else:
            # 格式 E1051218 → 105度12分18秒
            degrees = float(num_part[:2])
            minutes = float(num_part[2:4])
            seconds = float(num_part[4:])

        # 计算十进制经度
        decimal = degrees + minutes / 60 + seconds / 3600
    return decimal


def traj_plot_with_waypoints(traj_data, waypoints_data, callsign):
    """

    参数:
    traj_data: 包含轨迹数据的 DataFrame，列包括 'trackId', 'longitude', 'latitude'
    waypoints_data: 包含航路点数据的 DataFrame，列包括 'name', 'longitude', 'latitude'
    track_id: 要绘制轨迹的航班 trackId
    """
    # 筛选出指定 trackId 的数据
    track_data = traj_data[traj_data['callSign'] == callsign]

    # 提取轨迹的经纬度数据
    track_longitudes = track_data['longitude'].astype(float)
    track_latitudes = track_data['latitude'].astype(float)

    # 提取航路点的经纬度数据
    waypoint_names = waypoints_data['POINT']
    waypoint_longitudes = waypoints_data['longitude'].astype(float)
    waypoint_latitudes = waypoints_data['latitude'].astype(float)

    plt.figure(figsize=(10, 8))

    # 绘制轨迹线
    plt.plot(track_longitudes, track_latitudes, label=f'Track: {callsign}', color='blue', linewidth=2)

    # 绘制轨迹点
    plt.scatter(track_longitudes, track_latitudes, color='red', label='Track Points', s=20)

    # 绘制航路点
    plt.scatter(waypoint_longitudes, waypoint_latitudes, color='green', label='Waypoints', s=30)

    # 标注航路点名称
    for i, name in enumerate(waypoint_names):
        plt.text(waypoint_longitudes[i], waypoint_latitudes[i], name, fontsize=8, ha='right', va='bottom',
                 color='green')

    # 添加标题和标签
    plt.title(f'Track ID: {callsign}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()


# 定义文件路径
def prase_waypoint_route(file_path) :

    # 读取文件，逐行处理
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化列表来存储解析后的数据
    parsed_data = []

    # 解析每行
    for line in lines:
        # 去掉行首行尾的空格和换行符
        line = line.strip()

        # 分割航路名称和航路点
        parts = line.split(',', 1)  # 只分割第一个逗号
        route_name = parts[0]  # 航路名称
        route_points = parts[1].split(',') if len(parts) > 1 else []  # 航路点

        # 解析航路点的名称和经纬度
        points = []
        point_names = []
        for point in route_points:
            if point:  # 确保点不为空
                # 分割点的名称和坐标
                name = point.split('/')[0]  # 第一个斜杠之前的内容作为点名称
                coords = point.split('/')[-1]
                try:
                    # 分割纬度和经度
                    lat, lon = coords.split(' ')
                    # 转换为小数格式
                    latitude = coor_convert(lat)  #
                    longitude =  coor_convert(lon)  # '
                    points.append((longitude, latitude))  # 注意：LineString 需要 (经度, 纬度) 格式
                    point_names.append(name)
                except ValueError as e:
                    print(f"解析错误: {e}，在点: {point}")
                    continue
        # 将解析后的数据存储到列表中
        parsed_data.append({
            'roucode': route_name,
            'pointcoor': points,
            'pointlist': point_names
        })

   

    return gdf


def way_point_adj(sta_route):
    all_points = set()
    for points in sta_route['pointlist']:
        all_points.update(points)

    # 将点排序并创建索引
    all_points = sorted(all_points)
    point_index = {point: idx for idx, point in enumerate(all_points)}

    # 初始化邻接矩阵
    num_points = len(all_points)
    adjacency_matrix = np.zeros((num_points, num_points), dtype=int)

    # 填充邻接矩阵
    for point_list in sta_route['pointlist']:

        for i in range(len(point_list) - 1):
            # 获取当前点和下一个点的索引
            current_point = point_list[i]
            next_point = point_list[i + 1]
            # 在邻接矩阵中设置值为1
            adjacency_matrix[point_index[current_point], point_index[next_point]] = 1
            adjacency_matrix[point_index[next_point], point_index[current_point]] = 1

    # 将邻接矩阵转换为 DataFrame 以便查看
    adjacency_df = pd.DataFrame(adjacency_matrix, index=all_points, columns=all_points)
    return adjacency_df


def get_segment(nodes, a, b):
    """
    获取两个节点间的路段（自动识别方向）
    :param nodes: 节点列表，如 ['A','B','C','D']
    :param a: 第一个端点
    :param b: 第二个端点
    :return: 两点间的节点列表（包含端点）
    """
    try:
        i, j = nodes.index(a), nodes.index(b)
        return nodes[i:j+1] if i < j else nodes[i:j-1:-1]
    except ValueError:
        return []

def seg_converse(valid_points, valid_seg,seg_lst,route,sta_route,unkonw_lst):
    updated_route = route.copy()
    for i, info in enumerate(seg_lst):
        info_idx = route.index(info)
        info_route =sta_route.loc[sta_route['roucode'] == info, 'pointlist'].iloc[0]
        up_info = route[info_idx - 1]
        next_info = route[info_idx + 1]
        # 这里会出现好几种情况
        # 4. 如果前后是存在一个未知的点  那么就跳过  截取不了
        if up_info in unkonw_lst or next_info in unkonw_lst:
            print(f"警告：存在未知类型点{up_info}，{next_info}")
            continue
         # 1.前后两点都是航路点  那么直接截取
        elif up_info in valid_points and next_info in valid_points:
            add_node = get_segment(info_route, up_info ,  next_info)

#         2. 前后都是航段  那么求航段交点  再进行截取
        elif up_info in  valid_seg and next_info in valid_seg:
            up_route = sta_route.loc[sta_route['roucode'] ==up_info, 'pointlist'].iloc[0]
            next_route = sta_route.loc[sta_route['roucode'] == next_info, 'pointlist'].iloc[0]
            up_node=[p for p in up_route if p in set(info_route)]
            next_node=[p for p in next_route if p in set(info_route)]
            add_node = get_segment(info_route, up_node, next_node)
            if not up_node or not next_node:
                print(f"警告：航段 {up_info} 或 {next_info} 与{info}无交点")
#           3. 前后分别是 航路点  以及航段  那么需要求交点 根据航路点与交点进行截取
        elif (up_info in valid_points and next_info in valid_seg) or (up_info in valid_seg and next_info in valid_points):
            point = up_info if up_info in valid_points else next_info
            # 判断该航路点是不是再当前航路里面
            if point not in info_route:
                print(f"警告：航路点{point} 与当前航路{info}无交点")
                continue
            seg_route = sta_route.loc[sta_route['roucode'] == seg, 'pointlist'].iloc[0]
            seg = next_info if next_info in valid_seg else up_info
            common_point = [p for p in seg if p in set(info_route)]
            #  # 情况3a：上游是航路点，下游是航段
            if up_info in valid_points:
                # 判断该航路点是不是再当前航路里面
                add_node=get_segment(info_route, point, common_point)
                # 情况3b：上游是航段，下游是航路点
            elif up_info in valid_seg:
                add_node=get_segment(info_route, common_point,point)
        print(add_node)
        updated_route[updated_route.index(info)] = add_node

    return  updated_route


def find_invalid_points(fpl_data, all_points,sta_route):
    # 创建一个集合便于快速查找
    valid_points = set(all_points['point'])
    valid_seg=set(sta_route['roucode'])
    # 存储结果
    results = []

    # 遍历每一条飞行计划
    for idx, row in fpl_data.iterrows():
        date=row['etd'].date()
        flight_name = row['arcid']
        plantype=row['plantype']
        arctype=row['arctyp']
        wktrc=row['wktrc']
        cruisespd=row['cruisespd']
        cruisehit=row['cruisehit']
        route_name = row['routename']
        route = row['route']

        if pd.isna(row['route']).all():  # 如果route数组所有元素都是NA
            print(f'Skipping {flight_name} - route is all NA')
            continue

        # 找出不在all_points中的航点
        invalid_points = [p for p in route if p not in valid_points]
        # 再次判断这些无效点是不是在航路集合里
        seg_lst=[info for info  in invalid_points if info in valid_seg]
        unkown_points=[info for info  in invalid_points if info not in valid_seg]
        # 是的话将路段转为节点序列
        converse_route=seg_converse(valid_points, valid_seg,seg_lst,route,sta_route,unkown_points)

        if invalid_points:
            results.append({
                'date':date,
                'flight_name':flight_name,
                'plantype':plantype,
                'arctype':arctype,
                'wktrc':wktrc,
                'cruisespd':cruisespd,
                'cruisehit':cruisehit,
                'routename': route_name,
                'origin_route': route,
                'full_route':converse_route,
                'invalid_points': ', '.join( unkown_points),

            })

    # 转换为DataFrame
    result_df = pd.DataFrame(results)
    return result_df

def waypoint_converse( waypoint_data):
    #     将经纬度  度分秒转化为小数点后五位格式
    waypoint_data['lon'] = waypoint_data['longitude'].apply(lambda x: coor_convert(x))
    waypoint_data['lat'] = waypoint_data['latitude'].apply(lambda x: coor_convert(x))


    #     将航路点 以及轨迹经纬度  转化为UTM投影 =====================================

    #  定义 WGS84 坐标系和 UTM 48N 坐标系
    wgs84 = Proj(init='epsg:4326')  # WGS84 坐标系
    utm48n = Proj(init='epsg:32648')  # UTM 48N 坐标系
    # 使用 pyproj.Transformer 进行坐标转换
    transformer = Transformer.from_proj(wgs84, utm48n, always_xy=True)
    # 转换坐标
    lon_array = waypoint_data["lon"].values
    lat_array = waypoint_data["lat"].values
    x, y = transformer.transform(lon_array, lat_array)  # 直接传入数组
    waypoint_data["xAxis"] = x
    waypoint_data["yAxis"] = y
    return waypoint_data

def get_point_coordinates(point_list, point_set):
        coordinates = []
        for point in point_list:
            # 找到匹配的点
            match = point_set[point_set['point'] == point]
            if not match.empty:
                # 获取匹配点的经纬度
                lat = match['lat'].values[0]
                lon = match['lon'].values[0]
                coordinates.append((lon,lat))
            else:
                print(f'{point}没有找到匹配的点')
                coordinates.append((None, None))  # 如果没有找到匹配的点，添加 None
        return coordinates



if __name__=='__main__':

    # 1.综合航迹数据处理===================================

    folder_path= r"D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\SL_DCP1_sdi_all"  # 航迹日志文件夹路径

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            temp_df = pd.read_csv(file_path)  # 读取单个文件为 DataFrame
            traj_data= temp_df[( temp_df['ARR'] == 'ZUUU') | (temp_df['DEP'] == 'ZUUU')]
            traj_data.sort_values(by=['trackId', 'time'], inplace=True)
            # 删除重复行，确保 trackId 和 timestamp 的组合是唯一的
            traj_data.drop_duplicates(subset=['trackId', 'time', 'longitude', 'latitude'], keep='first', inplace=True)
            print('Df drop duplictate done,',  traj_data.shape)
            # traj_data = pd.concat([traj_data, temp_df], ignore_index=True)  # 合并到总 DataFrame 中    # 筛选出zuuu进离场航班
            # 保存进文件
            # 如果文件已存在，以追加模式写入数据
            csv_file_path = r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\zuuu_traj_0126-0225.csv'
            if os.path.exists(csv_file_path):
                traj_data.to_csv(csv_file_path, index=False, mode='a', header=False)
            else:
                # 如果文件不存在，以写入模式写入数据，并包含表头
                traj_data.to_csv(csv_file_path, index=False)
            print(f'Data written to CSV file: {csv_file_path}')
            del temp_df
            gc.collect()
  


    # 2.航路点 航路网数据处理== == == == == == == == == == == == == == == == == == == == ==
    column_names = ['point', 'latitude', 'longitude']
    waypoint_data = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\origin_data\points.csv', header=None,
                                names=column_names)
    waypoint_data=waypoint_converse(waypoint_data)
    waypoint_data.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\waypoint.csv',index=False)
   
    file_path = r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\origin_data\allroutes.csv'
    parsed_data= prase_waypoint_route(file_path)
 # 将解析后的数据转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(parsed_data, geometry=[LineString(d['pointcoor']) for d in parsed_data])
    gdf['pointcoor'] = gdf['pointcoor'].apply(lambda x: ','.join([f"{lon},{lat}" for lon, lat in x]))
    gdf['pointlist'] = gdf['pointlist'].apply(lambda x: ','.join(x))

    # 设置坐标参考系统为 WGS84
    gdf.set_crs(epsg=4326, inplace=True)

    # # 保存为 GeoJSON 文件
    # gdf.to_file(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\allroutes.geojson', driver='GeoJSON')
    gdf.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\waypoints_route.csv')
    # # 保存为 Shapefile 文件
    gdf.to_file(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\allroutes.shp', driver='ESRI Shapefile')



    # 3.系统进离场点和路径=========================
   
    sid_star_info = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\origin_data\ZUUU_sid_star_info.csv')
    sid_star_info["POINT_LIST"] = sid_star_info["POINTS"].str.split("/")
    sid_star_point=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\origin_data\ZUUU_sid_star_points.csv',names=['point','coor'])
    sid_star_point['latitude']=sid_star_point["coor"].str.split("E").str[0]
    sid_star_point['longitude'] ='E'+ sid_star_point["coor"].str.split("E").str[1]
    sid_star_point = waypoint_converse(sid_star_point)
    sid_star_point.to_csv(fr'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\sid_star_point.csv', index=False)
    sid_star_info['pointcoor']=sid_star_info["POINT_LIST"].apply(lambda x: get_point_coordinates(x,sid_star_point))
    # 删除type列为missed 的行
    sid_star_info = sid_star_info[sid_star_info['TYPE'] != 'MISSED']
    sid_star_info['geometry'] = sid_star_info['pointcoor'].apply(lambda x: LineString(x))
    # 保存  进离场文件
    #  重命名列以保持一致性（如果需要）
    # 例如，将 sid_star_info 中的 'POINT_LIST' 重命名为 'pointlist'
    sid_star_info = sid_star_info.rename(columns={'POINT_LIST': 'pointlist'})
    sid_star_info = sid_star_info.rename(columns={'NAME': 'roucode'})
    sid_star_info = sid_star_info.drop(columns=['POINTS'])
    sid_star_info.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\ZUUU_sid_star_route.csv')
    # # 转换为 SHP图 方便查看
    sid_star_gdf = gpd.GeoDataFrame(sid_star_info, geometry='geometry')
    # # 设置坐标参考系统（CRS），例如 WGS84
    sid_star_gdf.set_crs(epsg=4326, inplace=True)
    sid_star_gdf.to_file(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\zuuu_sidstar_route_shp\ZUUU_sid_star_route.shp',driver='ESRI Shapefile')


    # 合并 waypoint  和 fixpoint
    all_points= pd.concat([waypoint_data, sid_star_point]).sort_values(by='point')
    # 删除重复点
    all_points.drop_duplicates(subset='point',keep='last',inplace=True)

    all_points.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\all_points.csv')
    # 把航路网合并
    all_routes=pd.concat([sta_route, sid_star_info])
    all_routes.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\updated_data\all_routes.csv')




    #     飞行计划数据=====================================
    fpl_data=pd.read_excel(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250225_tab_fpl_bak.xlsx')
    #查看你plantype有哪些种类 ['OVERFLY' 'ARR' 'DRAG' 'TIP' 'DEP' 'TRACK' 'TAG']
    unique_platype = fpl_data['plantype'].unique()
    # 只保留 ARR DEP 类型
    zuuu_fpl= fpl_data[fpl_data['plantype'].isin(['ARR', 'DEP'])]
    zuuu_fpl['route']=zuuu_fpl['route'].str.replace(r'\bDCT\b', '', regex=True).str.strip().str.replace(' +', ' ', regex=True)
    # 将 route 列中的字符串分割为列表
    zuuu_fpl['route'] = zuuu_fpl['route'].str.split()
    # 对于每一行的route
    for index, row in  zuuu_fpl.iterrows():
        route = row['route']
        # 检查route是否为NaN
        if route is None or (isinstance(route, float) and np.isnan(route)):
            continue  # 如果是NaN，则跳过当前迭代
        # 修改异常值
        for i, info in enumerate(route):
            if '/' in info:
                route[i] = info.split('/')[0]
        zuuu_fpl.at[index, 'route'] = route

    df=find_invalid_points(zuuu_fpl, fix_point, fix_route)












