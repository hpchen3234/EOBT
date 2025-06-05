'''使用同一停机位-跑道组合的第20百分位作为估计滑行时间
畅通滑行时间是统计某机场所有航班的滑行时间，
将所有滑行时间的20百分位数定为该机场的畅通滑行时间，即该滑行时间小于80%航班的滑行时间，大于20%航班的滑行时间。 '''


import  numpy  as np
import  pandas as pd

if __name__=='__main__':
    # 读取飞行计划
    # =====读取飞行计划
    # flight_plan = pd.read_csv(r'D:\eobt_Project\data\ZUUU数据集\06-10depdata_delete1.csv')
    flight_plan=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\20250307-双流拷贝\zuuu_20250306_arrdep_fpl.csv')

    # 去除滑行路线为空的行
    flight_plan = flight_plan.dropna(subset=[ 'slipline'])
    # 对于离场航班  滑行时间=lintime -taxtime
    # 进场航班  滑行时间=ovetime-taxtime
    flight_plan['lintime'] = pd.to_datetime(flight_plan['lintime'])
    flight_plan['taxtime'] = pd.to_datetime(flight_plan['taxtime'])
    

    # 分成进离场
    arr_fpl=flight_plan[flight_plan['plantype']=='ARR']
    dep_fpl=flight_plan[flight_plan['plantype']=='DEP']
    
    # 计算时间差值（以秒为单位）
    dep_fpl['delta_time'] = (dep_fpl['lintime'] - dep_fpl['taxtime']).dt.total_seconds()

    # 去重重复停机位 滑行道  跑到
    unique_routes = dep_fpl[['depgate', 'slipline', 'deprwy']].drop_duplicates()
    unique_routes = unique_routes.astype(str)
    # 按照 'depgate' 列排序
    unique_routes = unique_routes.sort_values(by=['deprwy','depgate'])

    # 按照停机坪-跑到进行分类
    results = []
    grouped=dep_fpl.groupby(by=['depgate','deprwy'])
    for (depgate,deprwy),group in grouped:
            # 统计该停机坪-跑到组合出现的次数
            count = len(group)
            slipline=group.iloc[0]['slipline']
            time_lst =group['delta_time'].tolist()
            desc = group['delta_time'].describe()
            # 计算该航段的二十百分位对应的滑行时长
            # 计算该航段的第 20 百分位对应的滑行时长
            quantiles = group['delta_time'].quantile([0.10,0.20, 0.80,0.90])

            # 将结果存储到列表中
            results.append({
                'depgate': depgate,
                'deprwy': deprwy,
                'slipline':slipline,
                'count': count,
                'time_distribution': time_lst,
                'mean_time': desc['mean'],  # 平均值
                'std_time': desc['std'],   # 标准差
                'min_time': desc['min'],   # 最小值
                '25th_time': desc['25%'],  # 第 25 百分位
                '50th_time': desc['50%'],  # 第 50 百分位（中位数）
                '75th_time': desc['75%'],  # 第 75 百分位
                'max_time': desc['max'],   # 最大值
                '10th_time': quantiles[0.10],  # 第 20 百分位
                '90th_time': quantiles[0.90],  # 第 80 百分位
                '20th_time': quantiles[0.20],  # 第 20 百分位
                '80th_time': quantiles[0.80]   # 第 80 百分位
                })
          
    result_df = pd.DataFrame(results)

    result_df.to_csv(r'D:\eobt_Project\ZUUU\surface_traj_match\zuuu_20250306_dep_fpl_20th_percentile_time.csv')


    # 停机位分区域  根据二十百分位估计
    # 如果不同停机位对应同一跑道的情况下 二十百分位数是一样的 那么这些停机位就是同一块区域 增添一列停机位区域

    result_df['region'] = result_df.groupby(['deprwy', 'twentyth_time']).ngroup() + 1