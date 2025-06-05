# 从需要预测滑行时间的航班计划中提取计划滑行路线
# 再根据机场底图 将其转化为节点序列
import  numpy  as np
import  pandas as pd
import xml.etree.ElementTree as ET
import geopandas as gpd
from collections import defaultdict
from zuuu_airport import geopoint_trans
import matplotlib.pyplot as plt
import seaborn as sns



def parse_std_route(file_name):
    # 解析机场xml文件提取经纬度
    tree = ET.parse(file_name)
    # 解析XML字符串并返
    root = tree.getroot()
    extracted_info = []
    for row in root.iter('row'):
        nodes =row.findall('node')
        node_data={}
        for node in nodes:
            param=node.get('param')
            value=node.get('value')
            node_data[param]=value
        extracted_info.append(node_data)
    std_route = pd.DataFrame(extracted_info)
    return  std_route

def expand_range(value):
    if '-' in value:
        parts = value.split(',')
        expanded_range = []
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                expanded_range.extend([str(i) for i in range(int(start), int(end) + 1)])
            else:
                expanded_range.append(part)
        return expanded_range
    else:
        return [value]

def build_matrix(df):
    # 提取所有端点并去重，生成唯一编号
    unique_points = pd.concat([df['point_1'], df['point_2']]).drop_duplicates().reset_index(drop=True)
    # 创建一个点的编号映射
    point_index = {point: idx for idx, point in unique_points.items()}
    # 初始化邻接矩阵，矩阵大小为点的数量
    adj_matrix = np.zeros((len(unique_points), len(unique_points)))
    adj_dict = {point: [] for point in unique_points}

    # 填充邻接矩阵：如果两个点在同一条线段上，设置对应矩阵位置为1
    for _, row in df.iterrows():
        p1 = row['point_1']
        p2 = row['point_2']
        idx1 = point_index[p1]
        idx2 = point_index[p2]
        if row['direct']=='Both':
            adj_dict[p1].append(p2)
            adj_dict[p2].append(p1)
            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1  # 无向图的情况，矩阵是对称的
        elif row['direct']=='Backward':
            adj_dict[p2].append(p1)
            adj_matrix[idx2, idx1] = 1
        else:
            adj_dict[p1].append(p2)
            adj_matrix[idx1, idx2] = 1
        # 将节点名称作为索引和列标签
    adj_matrix= pd.DataFrame(adj_matrix, index=unique_points, columns=unique_points)

    return adj_matrix,adj_dict

def find_stand_start(p1,p2,adj_dict):
        if p1!='524'or p2!='524':
            if len(adj_dict[p1])==1:
                return [p1,p2]
            else:
                return [p2,p1]
        else:
            # 如果 p1 是 '524'，直接返回原始顺序
            return [p1, p2]

def taxi_route(info):
    direction = info['direction']
    segments = info['segments']
    if len(segments) == 1:
        for u,v in segments:
            ordered_path = [u,v]
    else:
        # 创建一个字典来记录每个节点的连接关系
        graph = defaultdict(list)
        # 遍历每一条边，构建图和统计每个节点的连接次数
        start_list=[]
        for start, end in segments :
            start_list.append(start)
            graph[start].append(end)
            graph[end].append(start)
            # 考虑到方向问题  起始节点的考虑只能在start列表里 找出
        start_node = None
        if direction=='Both':
            for node in graph:
                if len(graph[node]) == 1:
                    start_node = node
                    break
        else:
            for node in start_list:
                if len(graph[node]) == 1:
                    start_node = node
                    break

            # 遍历节点，形成有序路径
        ordered_path = []
        visited = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                ordered_path.append(node)
                # 将未访问的相邻节点加入栈
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
    return ordered_path

    #    继续 寻找 滑行道间的 交点

def find_taxi_int(slipline, taxi_path):
        # 解析滑行道序列
        parts = slipline.split()
        if len(parts) == 1:
            return None
        # 初始化交点列表
        intersections = []
        # 遍历每对相邻滑行道
        for taxiway1, taxiway2 in zip(parts, parts[1:]):
            # 获取每对滑行道的节点序列
            route1 = set(taxi_path[taxiway1]['route'])
            route2 = set(taxi_path[taxiway2]['route'])
            # 找到共同的节点
            common_nodes = route1 & route2
            # 如果有交点，记录下来
            if not common_nodes:
                #         如果没有交点 可能是与支路有交点    直接将滑行道上所有点进行匹配
                node1 = set(taxi_path[taxiway1]['nodes'])
                node2 = set(taxi_path[taxiway2]['nodes'])
                common_nodes = node1 & node2

            intersections.append((taxiway1, taxiway2, list(common_nodes)))
        return intersections

def find_rwy_int(plantype,deprwy, arrrwy,slipline, rwyin_seg,rwyout_seg, taxi_path):
        parts = slipline.split()
        if plantype=='DEP':
            rwy_taxi = parts[-1]
            rwyname=deprwy
            RWY_seg=rwyin_seg
            print('dep',rwy_taxi,rwyname)

        else:
            rwy_taxi = parts[0] 
            rwyname=arrrwy
            RWY_seg=rwyout_seg
            print('arr',rwy_taxi,rwyname)
        rwy_path = taxi_path.get(rwy_taxi, None)  # 防止 KeyError
        print(rwy_path)
        # 两种情况
        # 1. 跑道和滑行道在 rwyin_seg 里有相应的路段
        rwy_seg = RWY_seg.loc[( RWY_seg['taxiname'] == rwy_taxi) &
                                ( RWY_seg['typename'] == rwyname), ['point_1', 'point_2']]
        print(rwy_seg)
        if not rwy_seg.empty:
            # 找出路径终点，也就是邻接关系只有一个的点
            end_node = rwy_seg['point_2'].values[0]
            int_node = rwy_seg['point_1'].values[0]
            return pd.Series([int_node, end_node])  # 转换为字典列表

        # 2. 没有相应的，则遍历进行匹配，求交叉点
        else:
            rwy =  RWY_seg[ RWY_seg['typename'] == rwyname]
            # 遍历跑道的每个路段，寻找与滑行道的交点
            for index, row in rwy.iterrows():
                point_1 = row['point_1']
                point_2 = row['point_2']
                # 检查滑行道路径是否与跑道路段有交点
                if point_1 in  rwy_path:
                    int_node = point_1
                    end_node = point_2
                    match_seg = [point_1, point_2]
                    return pd.Series([int_node, end_node])  # 返回列表

            # 如果没有找到交点，返回空值
            return pd.Series([ None, None])


def find_nearest_node(intersection_nodes,first_route,next_int):
        nearest_node = None
        min_index_diff = float('inf')  # 设定初始最小索引差值
        for node in intersection_nodes:
            index_diff = abs(first_route.index(node) - first_route.index(next_int))  # 计算索引差值
            if index_diff < min_index_diff:
                min_index_diff = index_diff
                nearest_node = node
        return nearest_node

def branch_node(node1,branch,node2,route):
        # 使用列表推导式找到含有 int_node 的路段
        # 这里有涉及到匹配到多个干支
        matching_branches = [segment for segment in branch if node1 in segment]
        print(matching_branches)
        adj_nodes = []
        for u, v in matching_branches:
            adj_node = u if u != node1 else v
            adj_nodes.append(adj_node)
        adj_node = find_nearest_node(adj_nodes, route, node2)

        return adj_node

def idx_order(start_index,end_index,route):
        if start_index <= end_index:
            # 如果 origin_int 在 int_node 前面，按正向截取
            segment = route[start_index:end_index + 1]
            # print('1:', segment)
        else:
            # 如果 origin_int索引 在 int_node 后面，按反向截取
            segment = list(reversed(route[end_index:start_index + 1]))
        return segment

def seg_range(origin_int,int_node,route,branch ):
        # 两交点间形成的路段  分成三种情况
        # 1.都在主干上
        if origin_int in route and int_node in route:
            # print(1)
            # 找到交点 origin_int 和 int_node 的索引
            start_index = route.index(origin_int)
            end_index = route.index(int_node)
            # 根据索引顺序截取路段
            segment=idx_order(start_index,end_index,route)
                # print('1:',segment)
        elif origin_int not in route and int_node in route:
            # print(2)
            adj_node = branch_node(origin_int, branch, int_node, route)
            branch_seg = [origin_int, adj_node]
            start_index = route.index(adj_node)
            end_index = route.index(int_node)
            segment = idx_order(start_index, end_index, route)
            branch_seg.extend(segment)
            segment = branch_seg
            segment = list(dict.fromkeys(segment))
            # print('2:',segment)
        else:
            # print(3)
            adj_node = branch_node(int_node, branch, origin_int, route)
            branch_seg = [adj_node, int_node]
            start_index = route.index(origin_int)
            end_index = route.index(adj_node)
            segment=idx_order(start_index,end_index,route)
            segment.extend(branch_seg)
            segment = list(dict.fromkeys(segment))
            # print('3:', segment)
        return segment

def rwy_taxi_route(end_node,rwy_int,taxi_int,taxi_path):
        # 倒序推
        print('taxi_int',taxi_int)
        print('end_node,rwy_int',end_node,rwy_int)
        if taxi_int==None:
            return [rwy_int,end_node]
        reverse_routes=[end_node]
        origin_int=rwy_int
        for item in reversed(taxi_int):
           taxiway2=item[1]
           int_node=item[2][0]
           route = taxi_path[taxiway2]['route']
           branch = taxi_path[taxiway2]['branch']
           print(taxiway2,origin_int,int_node)
           segment=seg_range(origin_int,int_node,route,branch)
           reverse_routes.extend(segment)
           origin_int = int_node

        reverse_routes = list(dict.fromkeys(reverse_routes))
        rwy_taxi_route=list(reversed(reverse_routes))
        return rwy_taxi_route

def find_node_recursive(node, route, adj_dict, visited=None, path=None, max_depth=10):
    """
    递归查找与 node 相连的节点，直到找到所有在 route 中的节点。
    :param node: 当前节点
    :param route: 滑行道主干路径
    :param adj_dict: 邻接字典，存储每个节点的邻点
    :param visited: 已访问的节点集合，防止重复访问
    :param path: 当前路径，记录递归过程中访问的所有节点
    :param max_depth: 最大递归深度，防止无限递归
    :return: 所有在 route 中的节点及其路径
    """
    if visited is None:
        visited = set()
    if path is None:
        path = []

    # 将当前节点添加到路径中
    path.append(node)

    results = []

    # 检查当前节点的邻点
    neighbors = adj_dict.get(node, [])
    intersection_nodes = [neighbor for neighbor in neighbors if neighbor in route]

    # 如果当前节点有多个邻点在路径上，停止递归并返回这些交点及其路径
    if intersection_nodes:
        for intersection_node in intersection_nodes:
            # 确保路径包含 stand_end 和交点
            full_path = path + [intersection_node]
            results.append((intersection_node, full_path))
        return results

    if max_depth <= 0:
        return results

    visited.add(node)

    for neighbor in neighbors:
        if neighbor not in visited:
            # 递归查找邻点
            neighbor_results = find_node_recursive(neighbor, route, adj_dict, visited, path[:], max_depth - 1)
            results.extend(neighbor_results)

    return results

def find_stand_int(depgate,slipline,rwy_taxi_route,stand_data,taxi_path,adj_dict):
        # 提取停机位路段
        stand_seg=stand_data[stand_data['typename']==depgate]
        if stand_seg.empty:
            print(f"停机位 {depgate} 未找到")
            return pd.Series([None,None,None])

        start_node=stand_seg['point_1'].iloc[0]
        stand_end = stand_seg['point_2'].iloc[0]
        parts = slipline.split()  # 按空格分割字符串
        first_letter = parts[0]  # 提取第一个部分的第一个字符
        # 主干  支干
        first_branch = taxi_path[first_letter]['branch']
        first_route = taxi_path[first_letter]['route']
        next_int = rwy_taxi_route[0]
        print(depgate,slipline,first_letter,next_int,start_node,stand_end)

        #  从停机坪处开始  找出停机坪与滑行路线的第一个滑行道的交点
        intersection_nodes = []
        if stand_end in first_route:
            print(0)
            first_int = stand_end
            start_route = [start_node, stand_end]
            node_seg = seg_range(first_int, next_int, first_route, first_branch)
            start_route.extend(node_seg)
            return pd.Series([start_route, start_node, first_int])
        else:
            #     若无直接交点证明停机坪与滑行道间还有一段甚至多段（不属于滑行道主干\不属于滑行道）的线段相连， 这里遍历的是主干
                #     找出滑行道与停机坪点间具有连接关系的点 可能有多个 这里可以结合滑行道的下一个交点进行 判断 
            print(1)
            intersection_nodes_with_paths= find_node_recursive(stand_end, first_route, adj_dict, max_depth=10)
            intersection_nodes=[node for node, path in intersection_nodes_with_paths]
            if intersection_nodes:
                    print('1-1')
                    first_int = find_nearest_node(intersection_nodes, first_route, next_int)
                    nearest_path = next(path for node, path in intersection_nodes_with_paths if node ==  first_int )
                    start_route = [start_node]+nearest_path
                    node_seg=seg_range(first_int,next_int,first_route,first_branch)
                    start_route.extend(node_seg)

                    return pd.Series([start_route, start_node, first_int])
     
                    #   主干中没有交点  但是支干中有交点 遍历支干
            # elif intersection_nodes == [] and first_branch != []:
            #         print('1-2')
            #         # 需要在支干中寻找
            #         for u, v in first_branch:
            #             if u in  adj_dict.get(stand_end, []) or  v in  adj_dict.get(stand_end, []) :
            #                 int_node = u if u in adj_dict.get(stand_end, []) else v
            #                 intersection_nodes.append(int_node)
            #         if len(intersection_nodes)==1:
            #             print(1)
            #             first_int=int_node
            #         else:
            #             first_int = find_nearest_node(intersection_nodes, first_route, next_int)
            #         start_route = [start_node, stand_end, first_int]
            #         node_seg=seg_range(first_int,next_int,first_route,first_branch)
            #         start_route.extend(node_seg)


def node_del(taxi_int, stand_int, rwy_int, route, adj_dict):
        #      遍历交点：除了taxi交点 还有stand_int rwy_int
        #    如果交点的前后两点有直接的链接关系  而且是弯道
        print(taxi_int, stand_int, rwy_int)
        int_nodes = [stand_int]
        if taxi_int:
            for taxi_1, taxi_2, node in taxi_int:
                int_nodes.extend(node)
                print(int_nodes)
        int_nodes.append(rwy_int)
        print(int_nodes)
        del_route = route.copy()
        for node in int_nodes:
            print(node)
            node_id = route.index(node)
            up_node = route[node_id - 1]
            next_node = route[node_id + 1]
            print(up_node, node, next_node)
            if next_node in adj_dict[up_node]:
                print(1)
                del_route.remove(node)
            else:
                print(2)
                continue
        return del_route

if __name__=='__main__':
    ''' # 解决std_route
    std_route=parse_std_route(r'D:\eobt_Project\ZUUU\zuuu-itwr_trp_std_route_online.xml')
    # 在固滑数据 中根据对应的停机位编号  提取路段节点链接关系
    #     首先将连字符扩展  601-611变为601 602.....611
    # 对 ENDNAME 列应用转换
    std_route['ENDNAME_EXPANDED'] = std_route['ENDNAME'].apply(expand_range)

    # =====读取飞行计划
    # flight_plan=pd.read_csv(r'D:\eobt_Project\data\ZUUU数据集\06-10depdata_delete1.csv')
    flight_plan=pd.read_excel(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250306_tab_fpl_bak.xlsx')
    # 只要进离场数据
    zuuu_fpl =  flight_plan[ flight_plan['plantype'].isin(['ARR', 'DEP'])]
    zuuu_fpl.to_csv(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250306_arrdep_fpl.csv')
    '''
    # 直接读取进离场的飞行计划

    zuuu_fpl=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250306_arrdep_fpl.csv')
    # 统计一个月每天每小时的航班架次    分析一下高峰流量时段：
    # 确保 timestamp 列是 datetime 类型
    zuuu_fpl['taxtime'] = pd.to_datetime(zuuu_fpl['taxtime'])
    # 提取日期和小时
    zuuu_fpl['date'] = zuuu_fpl['taxtime'].dt.date
    zuuu_fpl['hour'] = zuuu_fpl['taxtime'].dt.hour

    # 统计每天每小时的航班架次
    flight_counts = zuuu_fpl.groupby(['date', 'hour']).size().reset_index(name='flight_count')

    # 找到每天航班架次最多的小时段
    peak_hours = flight_counts.loc[flight_counts.groupby('date')['flight_count'].idxmax()]

    # 绘制每天每小时的航班架次
    plt.figure(figsize=(14, 8))
    sns.heatmap(flight_counts.pivot('hour', 'date', 'flight_count'), cmap='viridis', annot=True, fmt='d')
    plt.title('Daily Hourly Flight Counts')
    plt.xlabel('Date')
    plt.ylabel('Hour of Day')
    plt.show()

    # ===============将路线z
    # 去除滑行路线为空的行
    zuuu_fpl=zuuu_fpl.dropna(subset=['slipline'])   
    dep_fpl=zuuu_fpl[zuuu_fpl['plantype']=='DEP']
    # 去重重复停机位 滑行道  跑道
    # unique_routes = zuuu_fpl.drop_duplicates(subset=['arrgate', 'slipline', 'arrrwy'])
    unique_routes =dep_fpl.drop_duplicates(subset=['depgate', 'slipline', 'deprwy'])
    # unique_routes= unique_routes[].astype(str)
    unique_routes = unique_routes[~unique_routes['slipline'].str.contains(r'\bT4 C4\b')]

    # 读取机场滑行 数据

    zuuu_data=gpd.read_file(r'D:\eobt_Project\ZUUU\zuuu_shp\graphics.shp')
    zuuu_data = geopoint_trans(zuuu_data)
#     找到停机位对应的路段
    stand_data=zuuu_data[zuuu_data['type']=='stand']
    stand_data=stand_data[['typename','point_1','point_2']]
    # 前提是建立一个邻接矩阵 知道每个点的链接关系
    adj_matrix,adj_dict=build_matrix(zuuu_data)
    # 1 停机位序列（需要搞清谁是起点 除了点524 是两端都有链接  其余点都是只有一段链接的为起始点）
    # 将邻接点数量映射为一个字典
    adj_len_dict = {key: len(value) for key, value in adj_dict.items()}
    # 使用矢量化操作将邻接点数量映射到新列
    stand_data[['point_1', 'point_2']] = stand_data.apply(lambda row: find_stand_start(row['point_1'], row['point_2'], adj_dict),
                                            axis=1, result_type='expand')
    stand_data['typename'] = stand_data['typename'].astype(str)
    # # 这里是为了验证  初始节点的临街关系是不是都是1
    # stand_data['len1'] = stand_data['point_1'].map(adj_len_dict)
    # stand_data['len2'] = stand_data['point_2'].map(adj_len_dict)
    # 停机位序列已完成 与飞行计划中的停机位进行一个匹配

    # 找到跑道入口 路段 保留zuuu_data中 type 以‘rwy'字母开头的
    # rwy_seg=zuuu_data[zuuu_data['type'].str.startswith('rwy')]
    rwyin_seg = zuuu_data[zuuu_data['type']=='rwyin']
    rwyout_seg = zuuu_data[zuuu_data['type']=='rwyout']

    # ================  滑行路径转换 这里出现了滑行到编号包括主线以及支线（转弯路径 ） 无法很好的以一个序列进行表示
    # 因此分开  直线表示为主线 主线以序列表示 交叉口处的转弯路段表示为支线
    # 接下来 找节点序列 ：从停机位路段开始  找下一个链接的路段
    # # 一般顺序 由 stand-path-runway组成
    # node_squence=[]
    # # 1，,停机位匹配
    # plan_stand=flight_plan[['depgate']].drop_duplicates()
    # plan_stand= plan_stand.dropna()
    # plan_stand['depgate'] = plan_stand['depgate'].astype(str)
    # merge_df= plan_stand[['depgate']].merge(stand_data, left_on='depgate', right_on='typename', how='left')
    # # 建立 停机位字典
    # # 初始化停机位字典
    # stand_route = {}
    # # 遍历数据框，构建字典
    # for index, row in merge_df.iterrows():
    #     depgate = row['depgate']
    #     p1 = row['point_1']
    #     p2 = row['point_2']
    #     stand_route[depgate] = [p1, p2]
    # 检查是否有匹配不上的
    # nan_df = merge_df[merge_df.isna().any(axis=1)]
    # nan_df.sort_values(by='depgate')



    # 2.path 序列
    # 中包含多个滑行道交叉  思考怎么形成路径
    # 1.先提取出每一个滑行道的节点序列
    taxi_group=zuuu_data.groupby(by='taxiname')
    taxi_dict={}
    for taxiname, group in taxi_group:
        direct=group['direct'].unique()
        nodes = set(group['point_1']).union(set(group['point_2']))
        taxi_dict[taxiname] = {
            "direction": direct,
            'nodes':nodes,
            "segments": [],
            'branch':[]
        }
        for _, row in group.iterrows():
            if row['type']=='branch':
                taxi_dict[taxiname]['branch'].append((row['point_2'], row['point_1']))
                continue
            else:
                if row['direct'] == 'Backward':
                    taxi_dict[taxiname]['segments'].append((row['point_2'], row['point_1']))
                else:
                    taxi_dict[taxiname]['segments'].append((row['point_1'], row['point_2']))
    # 得出每个滑行道的节点序列
    taxi_path=taxi_dict
    for taxi,info in taxi_dict.items():
        taxi_path[taxi]['route']=taxi_route(info)



    unique_routes['taxi_int'] = unique_routes.apply(lambda row:
                                                    find_taxi_int(row['slipline'], taxi_path), axis=1)

    unique_routes[['rwy_int', 'rwy_end']] = unique_routes.apply(lambda row:
                                                                 find_rwy_int(row['plantype'],row['deprwy'],row['arrrwy'], row['slipline'], rwyin_seg,rwyout_seg,
                                                                              taxi_path), axis=1)

    # 删掉['rwy_int', 'rwy_end']为空的情况
    # 删除 ['rwy_int', 'rwy_end'] 均为空的行
    unique_routes = unique_routes.dropna(subset=['rwy_int', 'rwy_end'], how='all')
    unique_routes['rwy_taxi_route'] = unique_routes.apply(
        lambda row: rwy_taxi_route(row['rwy_end'], row['rwy_int'], row['taxi_int'], taxi_path), axis=1)


    unique_routes[['start_route','start_node','stand_int']] =  unique_routes.apply(
        lambda row: find_stand_int(row['depgate'], row['slipline'],row['rwy_taxi_route'],
                                   stand_data, taxi_path, adj_dict), axis=1 )
    # 删除['start_route','start_node','stand_int']的行（因为计划里的停机位在图中没有）
    unique_routes = unique_routes.dropna(subset=['start_route','start_node','stand_int'], how='all')

    unique_routes['route'] = unique_routes.apply(
        lambda row: row['start_route'] + row['rwy_taxi_route'], axis=1
    )
    unique_routes['route'] = unique_routes['route'].apply(lambda x: list(dict.fromkeys(x)))



    unique_routes['n_route'] = unique_routes.apply(
        lambda row:node_del(row['taxi_int'],row['stand_int'],row['rwy_int'],row['route'],adj_dict), axis=1 )

# 检查是否都有连接关系

#    存储一下  飞行计划路径节点序列
    unique_routes.to_csv('D:\eobt_Project\ZUUU\surface_traj_match\zuuu_250306_depfpl_route_node.csv')


#     转化一下标准路径的 节点序列












