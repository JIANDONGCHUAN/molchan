import numpy as np
import pandas as pd
import pygmt
from scipy.interpolate import RegularGridInterpolator
from gprm import PointDistributionOnSphere
from gprm.utils.proximity import contour_proximity, polyline_proximity, polygons_buffer, boundary_proximity, reconstruct_and_rasterize_polygons
from gprm.utils.create_gpml import gpml2gdf

# 新增：多线程依赖库
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from collections import OrderedDict


DEFAULT_DISTANCE_MAX = 1e7
DEFAULT_DISTANCE_STEP = 2e4
DEFAULT_GEOGRAPHIC_EXTENT = [-180., 180., -90., 90.]
DEFAULT_GEOGRAPHIC_SAMPLING = 0.25


def scipy_interpolater(da, points):
    f = RegularGridInterpolator((da.x.data, da.y.data), da.data.T, method='linear')
    return f(points)


def molchan_test(grid, 
                 points, 
                 distance_max=DEFAULT_DISTANCE_MAX, 
                 distance_step=DEFAULT_DISTANCE_STEP,
                 buffer_radius=1,
                 interpolater='pygmt'):
    """
    Molchan test for a set of points against one grid
    """
    
    earth_area = (6371.**2) * (4 * np.pi)
    
    # 计算每个点到目标的距离
    if interpolater == 'scipy':
        points['distance'] = scipy_interpolater(grid, points[['Longitude', 'Latitude']])
    else:
        points['distance'] = pygmt.grdtrack(
            points=pd.DataFrame(data=points[['Longitude', 'Latitude']]), 
            grid=grid, 
            no_skip=False, 
            interpolation='l',
            radius=buffer_radius,
            newcolname='dist'
        )['dist']
    
    # 计算目标距离栅格在每个等高线级别内的百分比
    grid_histogram = pygmt.grdvolume(
        grid, contour=[0, distance_max, distance_step], f='g', unit='k'
    )
    
    # 计算点的距离分布直方图
    (point_histogram, bin_edges) = np.histogram(
        points['distance'], 
        bins=np.arange(-distance_step, distance_max + distance_step, distance_step)
    )
    
    permissible_area = grid_histogram.iloc[0, 1]
    print(f'Total permissible area is {100 * permissible_area / earth_area:.1f}% of total Earth surface')
    
    # 计算栅格分数和点分数
    grid_fraction = 1 - grid_histogram.iloc[:, 1] / permissible_area
    points_fraction = 1 - np.cumsum(point_histogram) / len(points)
    
    # 计算技能评分
    Skill = 0.5 + np.trapz(grid_fraction, points_fraction)
    
    return grid_fraction[::-1], points_fraction[::-1], Skill


def molchan_point(grid, 
                  points, 
                  distance_max=DEFAULT_DISTANCE_MAX, 
                  distance_step=DEFAULT_DISTANCE_STEP, 
                  buffer_radius=1, 
                  interpolater='pygmt',
                  verbose=False,
                  return_fraction=True):
    """
    Molchan test for a single point
    """
    
    earth_area = (6371.**2) * (4 * np.pi)
    
    # 计算点到目标的距离
    if interpolater == 'scipy':
        points['distance'] = scipy_interpolater(grid, points[['Longitude', 'Latitude']])
    else:
        points['distance'] = pygmt.grdtrack(
            grid=grid,
            points=points[['Longitude', 'Latitude']], 
            no_skip=False, 
            interpolation='l',
            radius=buffer_radius,
            newcolname='dist'
        )['dist']
    
    if not return_fraction:
        return float(points['distance'].values)
    
    # 计算目标距离栅格的分布
    grid_histogram = pygmt.grdvolume(
        grid, contour=[0, distance_max, distance_step], 
        f='g', unit='k', verbose='e'
    )
    
    permissible_area = grid_histogram.iloc[0, 1]
    if verbose:
        print(f'Total permissible area is {100 * permissible_area / earth_area:.1f}% of total Earth surface')
    
    grid_fraction = 1 - grid_histogram.iloc[:, 1] / permissible_area 
    
    # 插值计算点对应的面积分数
    area_better_than_points = np.interp(
        points['distance'], grid_histogram.loc[:, 0], grid_histogram.loc[:, 1]
    )
    
    return float(points['distance'].values), float(1 - area_better_than_points / permissible_area)


def space_time_molchan_test(raster_dict, 
                            point_distances,
                            healpix_resolution=128,
                            distance_max=DEFAULT_DISTANCE_MAX, 
                            distance_step=DEFAULT_DISTANCE_STEP,
                            interpolater='pygmt'):
    """
    Given a raster sequence and a set of point distances already extracted from them, 
    compute a molchan test result where the grid fraction is summed over all rasters
    in the sequence
    """
    
    # 生成Healpix点分布
    hp = PointDistributionOnSphere(distribution_type='healpix', N=healpix_resolution)
    hp_dataframe = pd.DataFrame(data={'x': hp.longitude, 'y': hp.latitude})

    space_time_distances = []

    # 提取所有时间步的栅格距离数据
    for reconstruction_time in raster_dict.keys():
        if interpolater == 'scipy':
            smpl = scipy_interpolater(
                raster_dict[reconstruction_time], hp_dataframe
            )
            space_time_distances.extend(smpl[np.isfinite(smpl)].tolist())
        else:
            smpl = pygmt.grdtrack(
                grid=raster_dict[reconstruction_time], 
                points=hp_dataframe, 
                no_skip=False,
                interpolation='l',
                newcolname='distance'
            )
            space_time_distances.extend(smpl['distance'].dropna().tolist())

    # 计算栅格和点的距离分布直方图
    (hp_histogram, bin_edges) = np.histogram(
        space_time_distances, 
        bins=np.arange(-distance_step, distance_max + distance_step, distance_step)
    )

    (pm_histogram, bin_edges) = np.histogram(
        point_distances,
        bins=np.arange(-distance_step, distance_max + distance_step, distance_step)
    )

    # 计算分数和技能评分
    grid_fraction = np.cumsum(hp_histogram) / len(space_time_distances)
    point_fraction = 1 - np.cumsum(pm_histogram) / len(point_distances)
    
    Skill = np.trapz(1 - grid_fraction, 1 - point_fraction) - 0.5

    return grid_fraction, point_fraction, Skill


def combine_raster_sequences(raster_dict1, raster_dict2):
    """
    Given two raster sequences (dictionaries with coincident keys), 
    generate a new raster sequence that multiplies the coincident rasters
    from each sequence
    """
    
    raster_dict3 = OrderedDict()

    for key in raster_dict1.keys():
        raster_dict3[key] = raster_dict1[key] * raster_dict2[key]
        
    return raster_dict3


def space_time_distances(raster_dict, gdf, age_field_name='age', 
                         distance_max=DEFAULT_DISTANCE_MAX, 
                         distance_step=DEFAULT_DISTANCE_STEP, 
                         buffer_radius=1,
                         interpolater='pygmt'):
    """
    Computes the distances to targets reconstructed to their time of appearance 
    from a raster sequence of raster grids
    
    The input gdf is assumed to have reconstructed coordinates in its geometry
    """
    
    results = []

    for i, row in gdf.iterrows():
        reconstruction_time = row[age_field_name]
        result = molchan_point(
            raster_dict[reconstruction_time],
            pd.DataFrame(data={'Longitude': [row.geometry.x], 'Latitude': [row.geometry.y]}),
            distance_max=distance_max, 
            distance_step=distance_step, 
            buffer_radius=buffer_radius, 
            interpolater=interpolater
        )
        results.append(result)

    return pd.DataFrame(data=results, columns=['distance', 'area_fraction'])


def generate_raster_sequence_from_polygons(features,
                                           rotation_model,
                                           reconstruction_times,
                                           sampling=DEFAULT_GEOGRAPHIC_SAMPLING,
                                           buffer_distance=None,
                                           max_workers=None):  # 新增：max_workers参数
    """
    Given some reconstructable polygon features, generates a series of 
    raster grids (one per reconstruction time)
    """
    
    raster_dict = OrderedDict()

    # 内部函数：处理单个时间步的栅格生成（用于多线程）
    def _process_single_time(reconstruction_time):
        # 原单线程逻辑（不变）
        tmp = reconstruct_and_rasterize_polygons(
            features, rotation_model, reconstruction_time, sampling=sampling
        )
        tmp = tmp.where(tmp != 0, np.nan)
        
        if buffer_distance is not None:
            bn = boundary_proximity(tmp)
            tmp.data[bn.data <= buffer_distance] = 1
        
        return reconstruction_time, tmp  # 返回（时间步，对应栅格）

    # 多线程执行：并行处理所有时间步
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有时间步任务到线程池
        futures = [executor.submit(_process_single_time, t) for t in reconstruction_times]
        
        # 收集结果（按时间步顺序存入字典，保证输出顺序与输入一致）
        for future in as_completed(futures):
            t, raster = future.result()
            raster_dict[t] = raster
        
    return raster_dict


def generate_distance_raster_sequence(target_features, 
                                      reconstruction_model,
                                      reconstruction_times,
                                      sampling=DEFAULT_GEOGRAPHIC_SAMPLING,
                                      region=DEFAULT_GEOGRAPHIC_EXTENT,
                                      max_workers=None):  # 新增：max_workers参数
    """
    Generates a sequence of distance rasters (one per reconstruction time)
    from target polyline features
    """
    
    prox_grid_sequence = OrderedDict()

    # 内部函数：处理单个时间步的距离栅格生成（用于多线程）
    def _process_single_time(reconstruction_time):
        # 原单线程逻辑（修复原代码中tmp未定义的bug）
        if isinstance(target_features, dict):
            r_target_features = gpml2gdf(target_features[reconstruction_time])
        else:
            r_target_features = reconstruction_model.reconstruct(
                target_features, reconstruction_time, use_tempfile=False
            )
        
        # 生成距离栅格（无目标特征时生成空栅格，避免报错）
        if r_target_features is not None:
            prox_grid = polyline_proximity(
                r_target_features, spacing=sampling, region=region
            )
        else:
            # 生成与区域匹配的空栅格（用np.nan填充）
            lon = np.arange(region[0], region[1] + sampling, sampling)
            lat = np.arange(region[2], region[3] + sampling, sampling)
            prox_grid = np.ones((len(lat), len(lon))) * np.nan
        
        return reconstruction_time, prox_grid  # 返回（时间步，对应距离栅格）

    # 多线程执行：并行处理所有时间步
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有时间步任务到线程池
        futures = [executor.submit(_process_single_time, t) for t in reconstruction_times]
        
        # 收集结果（保证时间步顺序）
        for future in as_completed(futures):
            t, grid = future.result()
            prox_grid_sequence[t] = grid
        
    return prox_grid_sequence
