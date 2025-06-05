import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy  as np
from shapely.geometry import LineString
from geopy.distance import geodesic
from pyproj import Proj, transform
from shapely.geometry import Point
from ZUTF.code.read_airport_data import is_straight_or_curve,point_geodata,plot_airport

def geopoint_trans(gdf):
    # 2.处理输出所有节点以及相应的坐标
    gdf['point_1']=gdf['id'].str.extract(r'-tp.(\d+)-')
    gdf['point_2'] = gdf['id'].str.extract(r'(\d+)$')
    gdf['x1']=gdf.geometry.apply(lambda x: x.coords[0][0])
    gdf['y1']=gdf.geometry.apply(lambda x: x.coords[0][1])
    gdf['x2']=gdf.geometry.apply(lambda x: x.coords[-1][0])
    gdf['y2']=gdf.geometry.apply(lambda x: x.coords[-1][1])
    return  gdf


if __name__=='__main__':

    # 读取Shapefile
    zuuu_gdf = gpd.read_file(r'D:\eobt_Project\ZUUU\zuuu_shp\graphics.shp')


    zuuu_gdf['distance']=  zuuu_gdf.geometry.apply(lambda x:x.length)


    #1. 原始SHP数据 投影坐标为"WGS 84 / UTM zone 48N"    地理坐标系为WGS 84
    # 目前的数据形式是投影坐标下  需要准为地理坐标下的经纬度
    # 查看原始坐标系
    print(zuuu_gdf.crs)
     # EPSG:32648

     # 后来发现可以不用转换  因为轨迹信息也是UTM 投影
     # 转换为WGS 84地理坐标系（GCS_WGS_1984）
    zuuu_geo = zuuu_gdf.to_crs(epsg=4326)  # WGS 84的EPSG代码是4326
    # 保存一下
    zuuu_geo .to_file(r'D:\eobt_Project\ZUUU\zuuu_shp\zuuu_graphics_geo.shp')
    # 查看转换后的坐标系
    print(zuuu_geo.crs)
#     添加直弯特征：
    zuuu_geo['road_shape'] = zuuu_geo.geometry.apply(is_straight_or_curve)
#

#    解析各滑行路段对应节点名称 以及端点经纬度
    zuuu_geo=geopoint_trans(zuuu_geo)
    zuuu_gdf=geopoint_trans(zuuu_gdf)
    # plot_airport(zuuu_geo)
#     整理所有节点的信息 节点编号 经纬度  xy
    zuuu_node=point_geodata(zuuu_gdf,zuuu_geo)
    zuuu_node.to_csv(r'D:\eobt_Project\ZUUU\zuuu_shp\zuuu_node_del.csv',index=False)




