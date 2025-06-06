import  numpy as np
import pandas as pd 
from itertools import chain
import ast

if __name__=='__main__':
     # 读取飞行计划

        flight_plan = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250306_arrdep_fpl.csv')
        flight_plan['lintime'] = pd.to_datetime(flight_plan['lintime'])
        flight_plan['taxtime'] = pd.to_datetime(flight_plan['taxtime'])
        dep_fpl=flight_plan[flight_plan['plantype']=='DEP']
        # 计算时间差值（以秒为单位）
        dep_fpl['delta_time'] = ( dep_fpl['lintime'] -  dep_fpl['taxtime']).dt.total_seconds()

        # 筛选重要列
        dep_fpl= dep_fpl[['plantype', 'arcid', 'arctyp', 'wktrc',  'depgate', 'deprwy', 'slipline', 'delta_time']]
        

        # 读取滑行路线对应的节点序列
        fp_route=pd.read_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\zuuu_250306_depfpl_route_node.csv')
        # 读取机场图结构文件
        zuuu_gdf = pd.read_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\zuuu_gragh.csv')
        # 这里需填充一下空值
        # 对于typename 里的空值 填充为’taxiway'
        zuuu_gdf['typename'].fillna('taxiway', inplace=True)
        # 对于taxiname的空值 填充旁边'typename'列中的内容
        zuuu_gdf['taxiname'].fillna(zuuu_gdf['typename'], inplace=True)
    # 将 airport_graph 中的滑行道信息转换为更高效的查找结构

        taxiway_lookup = {
                frozenset([row['point_1'], row['point_2']]): {
                        'type':row['type'],                        
                        'typename':row['typename'],
                        'taxiname': row['taxiname'],
                        'direct':row['direct'],
                        'seg_type': row['road_shape'],
                        'distance': row['distance']
                }
                for _, row in zuuu_gdf.iterrows()
        }


         # 读取之前统计的精细滑行区间数据库
        fine_result=pd.read_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\fine_result_df.csv')
        # 对于这个结果  添加type说明
        fine_result['type'] = fine_result.apply(lambda row: taxiway_lookup[frozenset([row['node_1'], row['node_2']])]['type'], axis=1)
        
        # 对数据库进行划分 不要这么细
       # 定义新的分组键
        group_keys = ['flight_type', 'type', 'typename', 'taxiname', 'seg_type', 'direct']
        # 重新分组并聚合timelst
        fine_result['taxitime_distribution'] = fine_result['taxitime_distribution'].apply(ast.literal_eval)
        fine_result['speed_distribution'] = fine_result['speed_distribution'].apply(ast.literal_eval)

        coarse_results_df = fine_result.groupby(group_keys).agg({'taxitime_distribution': lambda x: list(chain.from_iterable(x.tolist())), 
                                                                 'speed_distribution': lambda x: list(chain.from_iterable(x.tolist()))}).reset_index() 

        coarse_results_df['taxitime_max'] = coarse_results_df['taxitime_distribution'].apply(lambda x: np.max(x))
        coarse_results_df['taxitime_min'] = coarse_results_df['taxitime_distribution'].apply(lambda x: np.min(x))
        coarse_results_df['taxitime_mean'] = coarse_results_df['taxitime_distribution'].apply(lambda x: np.mean(x))
        coarse_results_df['taxitime_25%'] =coarse_results_df['taxitime_distribution'].apply(lambda x: np.percentile(x, 25))
        coarse_results_df['taxitime_75%'] = coarse_results_df['taxitime_distribution'].apply(lambda x: np.percentile(x, 75))

        coarse_results_df['speed_mean'] = coarse_results_df['speed_distribution'].apply(lambda x: np.mean(x))
#          计算不同route  对应的滑行时间
#         创建空列表：
        fp_result = []
        for idx, row in fp_route.iterrows():  # 使用 iterrows() 迭代每一行
            flight_type = 'dep'
            depgate=row['depgate']
            slipline=row['slipline']
            deprwy=row['deprwy']
            route_list =ast.literal_eval(row['n_route'])  # 假设 n_route 是一个列表
                  
            # 遍历 route_list 中的每两个相邻点
            for n1, n2 in zip(route_list, route_list[1:]):
                n1,n2=int(n1),int(n2)
                # 在这里处理每对点 n1 和 n2
                #     滑行道需要在机场数据中寻找n1,n2 对应的滑行道
                if frozenset([n1, n2]) in taxiway_lookup:
                    print('True')
                    type=taxiway_lookup[frozenset([n1, n2])]['type']
                    typename=taxiway_lookup[frozenset([n1, n2])]['typename']
                    taxiname=taxiway_lookup[frozenset([n1, n2])]['taxiname']
                    seg_type=taxiway_lookup[frozenset([n1, n2])]['seg_type']
                    direct=taxiway_lookup[frozenset([n1, n2])]['direct']
                    seg_dis=taxiway_lookup[frozenset([n1, n2])]['distance']

                    # 那么该路段上的平均速度是
                    #     1 简单统计法 就是 同一滑行道 同一直弯类型上的平均速度
                    coarse_speed= coarse_results_df[
                         (coarse_results_df['type'] == type)&  
                        (coarse_results_df['typename'] == typename) &
                        (coarse_results_df['taxiname'] == taxiname) &
                        (coarse_results_df['seg_type'] == seg_type) &
                        (coarse_results_df['direct'] == direct)]['speed_mean']
                    coarse_time=coarse_results_df[
                         (coarse_results_df['type'] == type)&  
                        (coarse_results_df['typename'] == typename) &
                        (coarse_results_df['taxiname'] == taxiname) &
                        (coarse_results_df['seg_type'] == seg_type) &
                        (coarse_results_df['direct'] == direct)]['taxitime_mean']
                            # 出现没有的情况
                    if coarse_speed.empty:
                        coarse_speed = None
                    else:
                        coarse_speed = coarse_speed.item()  # 如果不为空，取值
                    if coarse_time.empty:
                          coarse_time=None
                    else:
                           coarse_time = coarse_time.item()
                    # 计算 taxitime
                    taxitime = seg_dis / coarse_speed if coarse_speed is not None else None

                    fp_result.append({'flight_type': flight_type,
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
                                         'avg_time': coarse_time ,
                                        'taxi_time':taxitime
                                })
        fp_result_df = pd.DataFrame(fp_result)
        fp_result_df.to_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\fine_result_df.csv')


  
#         2.读取统计的20th
        twentyth_taxitime=pd.read_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\zuuu_20250306_dep_fpl_20th_percentile_time.csv')
#         将结果中的停机坪与跑道 与twentyth_taxitime的停机坪与跑道列对应  匹配twentyth_taxitime中的滑行时间

        fp_result_df = fp_result_df.merge(
                twentyth_taxitime,
                on=['depgate', 'deprwy'],  # 按照停机坪和跑道进行匹配
                how='left'  # 使用左连接，确保 fp_result_df 中的所有行都被保留
        )
# 删除重复列 
        suffixes = [col for col in fp_result_df.columns if col.endswith('_x') or col.endswith('_y')]

        # 如果有重复列，删除 _y 列（保留 _x 列）
        for col in suffixes:
                if col.endswith('_x'):
                        original_col = col[:-2]  # 去掉 _x 后缀
                        if original_col + '_y' in fp_result_df.columns:
                                fp_result_df = fp_result_df.drop(columns=[original_col + '_y'])
                                # 将 _x 列重命名为原始列名
                                fp_result_df = fp_result_df.rename(columns={col: original_col})
                else:
                        continue
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
                        # # 定义在跑道上的一段 速度为3.8m/s
                        # last_seg= group.iloc[-1]['seg_dis']/3.8
                        statistic_taxitime = group['taxi_time'].sum() 
                        # +last_seg
                # 合并条件：满足任意一个条件时进行累加
                if (all_rows_are_not_null) or \
                        (first_row_is_null and group.iloc[1:]['taxi_time'].notna().all()) or \
                        (first_row_is_null and last_row_is_null and middle_rows_are_not_null):
 
                        flight_type=group.iloc[0]['flight_type']
                        slipline=group.iloc[0]['slipline']
                        twentyth_time=group.iloc[0]['25th_time']
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
   