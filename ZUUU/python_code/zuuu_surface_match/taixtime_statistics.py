# 已知轨迹对应的滑行路径结点序列
import  pandas as pd
import ast
from ZUUU.python_code.flightplan_route import build_matrix
from datetime import datetime, timedelta
import geopandas as gpd
import os
if __name__=='__main__':

        # 读取历史 路段上的时间统计数据
        folder_path=r'D:\eobt_Project\ZUUU\traj_match'
        # 将一个月的合并起来  再分类统计
        start_date = datetime(2025, 2, 3)
        end_date = datetime(2025, 3, 5)
        date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        seg_sta = pd.DataFrame()

        # 遍历日期列表，读取对应的 CSV 文件
        for date in date_list:
                # 生成文件名
                file_name = f"traj_{date.strftime('%Y%m%d')}.csv"
                file_path = os.path.join(folder_path, file_name)

                # 检查文件是否存在
                if os.path.exists(file_path):
                        # 读取 CSV 文件
                        df = pd.read_csv(file_path)
                        # 将数据追加到 all_data DataFrame 中
                        seg_sta = pd.concat([seg_sta, df], ignore_index=True)
        seg_sta.mean_speed= seg_sta.mean_speed1
        seg_sta.drop(columns='mean_speed1',inplace=True )
        seg_sta['delta_t']=seg_sta['t2']-seg_sta['t1']



        airport_gragh=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_gragh.csv')
        # 这里需填充一下空值
        # 对于typename 里的空值 填充为’taxiway'
        airport_gragh['typename'].fillna('taxiway', inplace=True)
        # 对于taxiname的空值 填充旁边'typename'列中的内容
        airport_gragh['taxiname'].fillna(airport_gragh['typename'], inplace=True)

        adj_matrix, adj_dict = build_matrix(airport_gragh)
        # 对统计数据分类
        # 如航班类型 进离场、航段是属于哪个滑行道  航段是直行还是转弯段
        rwy_set=[397 ,570 ,614, 764, 835, 913 ,980 ,0,4 ,12 ,22, 31 ,30,
                 40 ,50,58, 71 ,1050 ,91]

        # 1.首先判断进离场类型
        updated_groups = []

        # 对统计数据分类
        for (trackid, date), group in seg_sta.groupby(by=['trackid', 'segment_id']):
                # 判断进离场类型

                n1=group.iloc[0]['node_1']
                n2=group.iloc[-1]['node_2']
                if n1 in rwy_set or any(n1 in adj_dict[rwy] for rwy in rwy_set ):
                        group['fl_type'] = 'arr'
                elif n2 in rwy_set or any(node in rwy_set for node in adj_dict[n2]):
                        group['fl_type'] = 'dep'
                else:
                        # 代表轨迹覆盖不完全  不能呈现一条完整的从停机位到跑道的路径
                        group['fl_type'] = 'unknown'

                # 将修改后的分组添加到列表中
                updated_groups.append(group)

        # 将所有修改后的分组合并回一个 DataFrame
        seg_sta = pd.concat(updated_groups, ignore_index=True)


        # 2.航段属于哪个滑行道
        seg_sta[['taxiname','typename','direct']] = None  # 添加一个新列用于存储滑行道名称

        # 将 airport_graph 中的滑行道信息转换为更高效的查找结构

        taxiway_lookup = {
                frozenset([row['point_1'], row['point_2']]): {
                        'typename':row['typename'],
                        'taxiname': row['taxiname'],
                        'direct':row['direct'],
                        'seg_type': row['road_shape'],
                        'distance': row['distance']
                }
                for _, row in airport_gragh.iterrows()
        }

        # 遍历 seg_sta 中的每对节点
        for idx, row in seg_sta.iterrows():
                n1 = row['node_1']
                n2 = row['node_2']

                # 检查 [n1, n2] 是否在机场图中，不考虑顺序
                if frozenset([n1, n2]) in taxiway_lookup:
                        # 如果属于滑行道，获取对应的信息
                        seg_sta.at[idx, 'taxiname'] = taxiway_lookup[frozenset([n1, n2])]['taxiname']
                        seg_sta.at[idx, 'typename'] = taxiway_lookup[frozenset([n1, n2])]['typename']
                        seg_sta.at[idx, 'direct'] = taxiway_lookup[frozenset([n1, n2])]['direct']



        # 根据分类特征进行统计
        # 首先筛除 速度特别小的航段  一般都是有航空器停留 的情况
        seg_sta=seg_sta[seg_sta['mean_speed']>=2]

        # 分组统计
        # 按两种情况分   1， 不细致到每一个路段端点  而是根据路段类型（是停机位 跑道 还是普通滑行道），
        # 滑行道名称（各类滑行道名称 EFT...）  滑行道直弯类型 ，滑行方向 进行划分
        # 粗分结果
        coarse_results = []
        # 细分则是在此基础上  对端点进行分类  要求端点一模一样的路段上的速度分布情况
        fine_results=[]
        grouped=seg_sta.groupby(by=['fl_type','typename','taxiname','seg_type','direct'])
        for (flight_type,typename,taxiname,seg_type,direct),group in  grouped:
                if flight_type == 'dep':  # 只处理离场类型
                        print(flight_type,typename,taxiname,seg_type,direct)
                        count = len(group)
                        #
                        speed_lst = group['mean_speed'].tolist()
                        # 计算该航段的滑行速度平均值
                        avg_speed = group['mean_speed'].mean()
                        # 将结果存储到列表中
                        coarse_results.append({
                                'flight_type': flight_type,
                                'typename':typename,
                                'taxiname': taxiname,
                                'seg_type': seg_type,
                                'direct':direct,
                                'count': count,
                                'speed_distribution': speed_lst,
                                'avg_speed': avg_speed
                        })

                        # 对于每个分组，按 (node_1, node_2) 统计出现次数
                        grouped_by_nodes = group.groupby(by=['node_1', 'node_2'])

                        for (node_1, node_2), node_group in grouped_by_nodes:
                                # 统计该航段出现的次数
                                count = len(node_group)
                                #
                                speed_lst = node_group['mean_speed'].tolist()

                                # 计算该航段的滑行速度平均值
                                avg_speed = node_group['mean_speed'].mean()

                                # 将结果存储到列表中
                                fine_results.append({
                                        'flight_type': flight_type,
                                        'typename':typename,
                                        'taxiname': taxiname,
                                        'seg_type': seg_type,
                                        'direct':direct,
                                        'node_1': node_1,
                                        'node_2': node_2,
                                        'count': count,
                                        'speed_distribution':speed_lst,
                                        'avg_speed': avg_speed
                                })

                # 将结果转换为 DataFrame
                coarse_results_df=pd.DataFrame(coarse_results)
                fine_result_df = pd.DataFrame(fine_results)

        # 基于上述统计路段结果 对飞行计划路径 进行划分 匹配

        # 读取飞行计划
        flight_plan = pd.read_csv(r'D:\eobt_Project\data\ZUUU数据集\06-10depdata_delete1.csv')
        flight_plan['lintime'] = pd.to_datetime(flight_plan['lintime'])
        flight_plan['taxtime'] = pd.to_datetime(flight_plan['taxtime'])

        # 计算时间差值（以秒为单位）
        flight_plan['delta_time'] = (flight_plan['lintime'] - flight_plan['taxtime']).dt.total_seconds()

        # 筛选重要列
        flight_plan=flight_plan[['plantype', 'arcid', 'arctyp', 'wktrc',  'depgate', 'deprwy', 'slipline', 'delta_time']]

        fp_route=pd.read_csv(r'D:\eobt_Project\ZUUU\flight_plan_route_node.csv')
#          计算不同route  对应的滑行时间
#         创建空列表：
        fp_result = []
        for idx, row in fp_route.iterrows():  # 使用 iterrows() 迭代每一行
                flight_type = 'dep'
                depgate=row['depgate']
                slipline=row['slipline']
                deprwy=row['deprwy']

                route_list =ast.literal_eval(row['n_route'])  # 假设 n_route 是一个列表
                # 后续逻辑...
                # 遍历 route_list 中的每两个相邻点
                for n1, n2 in zip(route_list, route_list[1:]):
                    n1,n2=int(n1),int(n2)
                        # 在这里处理每对点 n1 和 n2
                    #     滑行道需要在机场数据中寻找n1,n2 对应的滑行道
                    if frozenset([n1, n2]) in taxiway_lookup:
                            print('True')
                            typename=taxiway_lookup[frozenset([n1, n2])]['typename']
                            taxiname=taxiway_lookup[frozenset([n1, n2])]['taxiname']
                            seg_type=taxiway_lookup[frozenset([n1, n2])]['seg_type']
                            direct=taxiway_lookup[frozenset([n1, n2])]['direct']
                            seg_dis=taxiway_lookup[frozenset([n1, n2])]['distance']

                    # 那么该路段上的平均速度是
                    #     1 简单统计法 就是 同一滑行道 同一直弯类型上的平均速度
                    #     2.精细 :必须每个路段对应  也就是两点是相同的   (问题在于数据覆盖不全  ,有的路段 统计数据中没有)
                            # 粗糙统计法
                            coarse_speed = coarse_results_df[
                                (coarse_results_df['typename'] == typename) &
                                (coarse_results_df['taxiname'] == taxiname) &
                                (coarse_results_df['seg_type'] == seg_type) &
                                (coarse_results_df['direct'] == direct)
                                ]['avg_speed']
                            # 出现没有的情况
                            if coarse_speed.empty:
                                    coarse_speed = None
                            else:
                                    coarse_speed = coarse_speed.item()  # 如果不为空，取值

                            # 计算 taxitime
                            taxitime = seg_dis / coarse_speed if coarse_speed is not None else None

                            fp_result.append({
                                        'flight_type': flight_type,
                                         'depgate':depgate,
                                        'slipline':slipline,
                                        'deprwy': deprwy ,
                                        'route':route_list ,
                                         'typename': typename,
                                        'taxiname': taxiname,
                                        'seg_type': seg_type,
                                        'direct': direct,
                                        'node_1': n1,
                                        'node_2': n2,
                                        'seg_dis':seg_dis,
                                        'avg_speed': coarse_speed ,
                                        'taxi_time':taxitime
                                })
        fp_result_df = pd.DataFrame(fp_result)



#         读取统计的20th
#         这里的二十百分位没有统计完全  因此普遍统计的时间较少  还是得根据轨迹来算比较真实

        twentyth_taxitime=pd.read_csv(r'D:\eobt_Project\ZUUU\ICAO_20th_percentile_time.csv')
#         将结果中的停机坪与跑道 与twentyth_taxitime的停机坪与跑道列对应  匹配twentyth_taxitime中的滑行时间

        result = fp_result_df.merge(
                twentyth_taxitime,
                on=['depgate', 'deprwy'],  # 按照停机坪和跑道进行匹配
                how='left'  # 使用左连接，确保 fp_result_df 中的所有行都被保留
        )

#        删除其他无用行
        fp_result_df=result.drop(columns=['Unnamed: 0',  'time_distribution'])



#         累加路段时间 需要判断有的路径没有匹配完全 所以时间是空值的
#         这里先对停机位 跑道进行分组   然后  只选该组第一个为空值 其他不为空值的计算时间累加
#         注意雷达信息没有涉及停机坪  都是靠拖车牵引而出，所以累加路段时不要加上停机坪的时间
        total_taxitime=[]
        for (depgate,deprwy),group in fp_result_df.groupby(by=['depgate','deprwy']):
                # 检查第一行和最后一行的 taxi_time 是否为空
                first_row_is_null = pd.isna(group.iloc[0]['taxi_time'])
                last_row_is_null = pd.isna(group.iloc[-1]['taxi_time'])
                # 检查中间行是否都不为空
                middle_rows_are_not_null = group.iloc[1:-1]['taxi_time'].notna().all()
                # 检查全组是否都不为空
                all_rows_are_not_null = group['taxi_time'].notna().all()

                # 根据条件设置路段描述（英文）
                if all_rows_are_not_null:
                        missing_section_description = "Full available"
                        statistic_taxitime = group['taxi_time'].sum()
                elif first_row_is_null and group.iloc[1:]['taxi_time'].notna().all():
                        missing_section_description = "Missing  depagate "
                        statistic_taxitime = group['taxi_time'].sum()
                elif first_row_is_null and last_row_is_null and middle_rows_are_not_null:
                        # 同上 自己定义在停机坪 以及跑道处的滑行速度
                        missing_section_description = "Missing depgate runway "
                        # 定义在跑道上的一段 速度为3.8m/s
                        last_seg= group.iloc[-1]['seg_dis']/3.8
                        statistic_taxitime = group['taxi_time'].sum() +last_seg
                # 合并条件：满足任意一个条件时进行累加
                if (all_rows_are_not_null) or \
                        (first_row_is_null and group.iloc[1:]['taxi_time'].notna().all()) or \
                        (first_row_is_null and last_row_is_null and middle_rows_are_not_null):

                        # statistic_taxitime=group['taxi_time'].sum()
                        flight_type=group.iloc[0]['flight_type']
                        slipline=group.iloc[0]['slipline']
                        twentyth_time=group.iloc[0]['twentyth_time']
                        count=group.iloc[0]['count']
                        total_taxitime .append({
                                'flight_type': flight_type,
                                'depgate': depgate,
                                'slipline': slipline,
                                'deprwy': deprwy,
                                'count':count,
                                'twentyth_taxitime': twentyth_time,
                               'statistic_taxitime':   statistic_taxitime,
                                'missing_section_description': missing_section_description

                        })
        total_taxitime= pd.DataFrame(total_taxitime)



# 结果分析
#删除二十百分位'twentyth_taxitime‘列中超出15*60=900的行 再进行统计
        total_taxitime=total_taxitime[total_taxitime['twentyth_taxitime']<900]
        total_taxitime['time_diff']=abs(total_taxitime['statistic_taxitime']-total_taxitime['twentyth_taxitime'])
#         平均时间误差为
        mean_time_diff=total_taxitime['time_diff'].mean()
        # 59.66090804174979




# 实际的滑行时间对比
        flight_plan = flight_plan.merge(total_taxitime[['depgate', 'deprwy', 'statistic_taxitime','twentyth_taxitime']],
                                        on=['depgate', 'deprwy'],
                                        how='left')

