import numpy  as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #     # 根据什么进行分类
# Index['date', 'callSign', 'connected', 'trajectory_idx', 'trajectory_idx_2',
#        'node_name', 'node_name_2', 'time', 'time_2', 'h0', 'h0_2', 'h', 'h_2',
#        'CFL', 'CFL_2', 'longitude', 'longitude_2', 'latitude', 'latitude_2',
#        'SPEED', 'SPEED_2', 'HEADING', 'HEADING_2', 'warnFlag', 'ARR', 'DEP',
#        'EOBT', 'RVSM', 'FR', 'acftTurb', 'distance', 'time_delta',
#        'mean_speed', 'plantype', 'arctyp', 'wktrc', 'cruisespd', 'cruisehit'],
#      
#     #   先根据'ARR', 'DEP', 'plantype', 'arctyp', 'wktrc', 'node_name', 'node_name_2' 划分
# 然乎对该分组下的   'h', 'h_2' 'CFL', 'CFL_2',进行一个统计 某些高度划在一起  如使用一些聚类算法# 定义计算真航线角的函数
def calculate_true_course(lat1, lon1, lat2, lon2):
     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
     delta_lon = lon2 - lon1   
     y = np.sin(delta_lon) * np.cos(lat2)  
     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
     tc = np.degrees(np.arctan2(y, x))
     return tc % 360  # 确保结果在0°到360°之间       
        

 

    


class FlightLevelMatcher():
    def __init__(self, df):
       
        self.df = df

        # （1）真航线角在0度至179度范围内，飞行高度由900米至8100米，每隔600米为一个高度层；飞行高度由8900至12500米，每隔600米为一个高度层；飞行高度12500米以上每隔1200米为一个高度层。

        # （2）真航线角在180度至359度范围内，飞行高度由600米至8400米，每隔600米为一个高度层；飞行高度9200米至12200米，每隔600米为一个高度层；飞行高度13100米以上，每隔1200米为一个高度层。

        #    计算最大值和最小值的差距

    def get_standard_height_layers(self, true_course):
        """
        根据飞行方向返回标准高度层列表。
        
        :return: 标准高度层列表
        """
        if 0 <= true_course < 180:  # 东向
            return [0]+ list(range(900, 8101, 600)) + list(range(8900, 12501, 600)) + list(range(12500, 40001, 1200))
        else:  # 西向
            return[0] + list(range(600, 8401, 600)) + list(range(9200, 12201, 600)) + list(range(13100, 40001, 1200))


    def match_height_to_layer(self, height, layers):
        """
        将当前高度与标准高度层进行匹配，找到最接近的高度层。
        :param height: 当前高度
        :param layers: 标准高度层列表
        :return: 最接近的高度层
        """

        closest_layer = min(layers, key=lambda x: abs(x - height))
        return closest_layer

    def process_data(self):
        """
        处理 DataFrame，添加飞行方向和匹配的高度层列。
        """
    
        # 获取标准高度层
        self.df['standard_height_layers'] = self.df['true_course'].apply(self.get_standard_height_layers)

        # 匹配当前高度到标准高度层
        self.df['matched_height_layer'] = self.df.apply(lambda row: self.match_height_to_layer(row['h'], row['standard_height_layers']), axis=1)
        self.df.drop(columns='standard_height_layers')

        return self.df

  


if __name__=='__main__':
    # 读取匹配结果数据 
    correct_match=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\match_result\correct_match_result.csv')
    all_match=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\match_result\all_match_result.csv')
    un_match=pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\match_result\unmatche_result.csv')


    macth_data=  all_match[all_match['connected']==True]

    macth_data.loc[macth_data['ARR'] == 'ZUUU', 'plantype'] = 'ARR'
    macth_data.loc[macth_data['DEP'] == 'ZUUU', 'plantype'] = 'DEP'
    # 然后删除 重复值
    macth_data.drop_duplicates(subset=['date', 'callSign', 'connected', 'trajectory_idx', 'trajectory_idx_2', 'node_name', 'node_name_2'],inplace=True)
    # 删除时间异常值  
    macth_data=macth_data[macth_data['time_delta']!=0]
    macth_data=macth_data[macth_data['time_delta']<6000]
    
    # 空中飞行速度   范围为 
    macth_data=macth_data[macth_data['mean_speed']>1]
    # 统计  航段 平均时间  pp
    # 高度层划分  这里 有的计划里没有巡航高度  根据自身的高度 怎么划分

    # 读取OurAirports的机场数据
    ourairports_df = pd.read_csv(r'D:\eobt_Project\ZUUU\zuuu_data\zuuu_flroute\airports.csv')

   
    # 建立机场代码到经纬度的映射字典
    airport_dict = {
        row['ident']: (row['latitude_deg'], row['longitude_deg'])
        for _, row in  ourairports_df .iterrows()
    }


    # 找出所有缺失的机场代码
    missing_airports = set(macth_data['DEP']).union(set(macth_data['ARR'])) - set(airport_dict.keys())
    # 手动补充缺失的机场经纬度信息
    missing_airports = {
        'ZPTC': (24.9383, 98.4858),  # 腾冲驼峰机场[^63^]
        'ZSJH': (30.65, 117.65),    # 池州九华山机场[^58^][^60^]
        'ZUBZ': (31.86, 106.73),    # 巴中恩阳机场[^67^]
        'ZUDR': (31.22, 107.47),    # 达州河市机场[^67^]
        'ZUPL': (23.03, 100.95)     # 普洱思茅机场[^67^]
    }
    # 更新airport_dict
    airport_dict.update(missing_airports)



    # 应用映射并计算真航线角
    macth_data['true_course'] = macth_data.apply(lambda row: calculate_true_course(
        *airport_dict[row['DEP']],
        *airport_dict[row['ARR']]
    ), axis=1)

    FL=FlightLevelMatcher(macth_data)
    new_macth_data=FL.process_data()

    # 初始化一个空的列表，用于存储每个分组的结果
    group_results = []
    #    根据 'ARR', 'DEP', 'plantype', 'arctyp', 'wktrc', 'node_name', 'node_name_2' 'matched_height_layer'划分。
    grouped=new_macth_data.groupby(by=[ 'ARR', 'DEP',  'arctyp', 'wktrc', 'node_name', 'node_name_2', 'matched_height_layer'])
    for info ,group in grouped:
        stats = group['time_delta'].describe(percentiles=[0.25, 0.5, 0.75])  # 默认包含 25%, 50%, 75% 的百分位数
 
        result = {
        'ARR': info[0],
        'DEP': info[1],
        'arctyp': info[2],
        'wktrc': info[3],
        'node_name': info[4],
        'node_name_2': info[5],
        'matched_height_layer': info[6],
        'count': stats['count'],
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        '25%': stats['25%'],
        '50%': stats['50%'],
        '75%': stats['75%'],
        'max': stats['max']
    }

        # 将结果添加到列表中
        group_results.append(result)
    # 将结果列表转换为 DataFrame
    result_df = pd.DataFrame(group_results)