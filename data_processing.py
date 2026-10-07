"""
数据处理模块
包含数据集类和数据预处理函数
"""

import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import xarray as xr
import numpy as np
import os
import glob
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from config import *

class WeeklyEC_S2S_Dataset(Dataset):
    def __init__(self, start_year=2001, end_year=2020, cache_dir=CACHE_DIR):
        """初始化数据集"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f'/3240101047/qiaoez/dataset_{start_year}_{end_year}_with_all_levels.pkl')

        if os.path.exists(cache_file):
            print(f"从缓存加载数据集: {cache_file}")
            self._load_from_cache(cache_file)
            return

        print("创建新数据集并缓存...")
        self._create_dataset(start_year, end_year, cache_file)

    def _create_dataset(self, start_year, end_year, cache_file):
        """创建数据集并保存到缓存"""
        # 区域设置
        lat_range = LAT_RANGE
        lon_range = LON_RANGE

        # 初始化EC网格信息
        self.ec_latitudes = None
        self.ec_longitudes = None

        # 存储样本
        self.data = []
        self.dates = []
        self.lead_times = []
        self.years = []

        # 获取所有大气变量文件（200/500hPa + 850hPa数据均在相同文件中）
        gh_files = glob.glob(DATA_PATHS['gh_files'])
        pf_files = glob.glob(DATA_PATHS['pf_files'])
        pf13012_files = glob.glob(DATA_PATHS['pf13012_files'])
        tp_files = glob.glob(DATA_PATHS['tp_files'])

        print(f"找到 {len(gh_files)} 个gh文件, {len(pf_files)} 个pf文件, {len(pf13012_files)} 个pf13012文件")
        print(f"找到 {len(tp_files)} 个tp文件")

        # 按文件名排序确保匹配
        gh_files.sort()
        pf_files.sort()
        pf13012_files.sort()
        tp_files.sort()

        # CN05.1数据
        ds_cn = xr.open_dataset(DATA_PATHS['cn05_file'])
        ds_cn = ds_cn.sel(latitude=lat_range, longitude=lon_range)
        cn_times = pd.to_datetime(ds_cn['time'].values)

        # 1. 识别CN05.1中始终为NaN的经纬度位置
        valid_mask = ds_cn['pre'].notnull().any(dim='time')
        cn_valid_mask = valid_mask.values
        self.cn_lat = ds_cn.latitude.values
        self.cn_lon = ds_cn.longitude.values
        print(f"CN05.1网格大小: lat={len(ds_cn.latitude)}, lon={len(ds_cn.longitude)}")

        # 2. 创建青藏高原掩膜（可选，这里保留代码但注释掉实际应用）
        print("创建青藏高原掩膜...")
        tibet_mask = self._create_tibet_mask(ds_cn.latitude.values, ds_cn.longitude.values)

        # 3. 合并掩膜
        #self.cn_valid_mask = cn_valid_mask & tibet_mask
        self.cn_valid_mask = cn_valid_mask  # 只保留 CN05.1 有效掩膜
        print(f"合并掩膜统计 - CN05.1有效点: {np.sum(cn_valid_mask)}, 青藏高原内点: {np.sum(tibet_mask)}, 最终有效点: {np.sum(self.cn_valid_mask)}")

        # 加载地形高度数据
        print("加载地形高度数据...")
        ds_dem = xr.open_dataset(DATA_PATHS['dem_file'])
        self.dem_data = ds_dem['dem'].sel(latitude=lat_range, longitude=lon_range)
        print(f"地形高度数据加载完成，形状: {self.dem_data.shape}")

        # 初始化各通道最大值
        self.channel_max = None

        # 处理每个文件
        for i, tp_file in enumerate(tp_files):
            print(f"\n处理文件 {i+1}/{len(tp_files)}: {os.path.basename(tp_file)}")

            # 打开所有相关文件（不再有300hPa专用文件）
            ds_tp = xr.open_dataset(tp_file)
            ds_gh = xr.open_dataset(gh_files[i]) if i < len(gh_files) else None
            ds_pf = xr.open_dataset(pf_files[i]) if i < len(pf_files) else None
            ds_pf13012 = xr.open_dataset(pf13012_files[i]) if i < len(pf13012_files) else None

            # 时间变量
            times = pd.to_datetime(ds_tp['time'].values)

            # 检查时间长度是否为20*28=560天
            if len(times) != 560:
                print(f"文件 {tp_file} 包含 {len(times)} 天数据，非560天，跳过")
                ds_tp.close()
                if ds_gh: ds_gh.close()
                if ds_pf: ds_pf.close()
                if ds_pf13012: ds_pf13012.close()
                continue

            # 选择区域
            tp_data = ds_tp['tp'].sel(latitude=lat_range, longitude=lon_range)

            # 获取EC网格信息
            if self.ec_latitudes is None:
                self.ec_latitudes = tp_data.latitude.values
                self.ec_longitudes = tp_data.longitude.values
                print(f"EC网格尺寸: 纬度={len(self.ec_latitudes)}, 经度={len(self.ec_longitudes)}")

            # 按年分组 (20年)
            for year_idx, year in enumerate(range(start_year, end_year+1)):
                # 提取该年的所有日期 (每年28天)
                start_idx = year_idx * 28
                end_idx = start_idx + 28
                year_dates = times[start_idx:end_idx]

                # 保存原始日期（用于大气变量）
                original_dates = year_dates

                # 差分后将EC起始日期提前一天以匹配CN05.1数据
                adjusted_dates = [d - timedelta(days=1) for d in year_dates]

                # 提取该年的累计降水数据 (所有成员)
                year_tp = tp_data.sel(time=year_dates)

                # 方法: 对时间维度做差分，第一天的日降水=第一天的累计降水
                daily_precip = np.zeros_like(year_tp.values)
                # 对每个成员单独处理
                for n in range(year_tp.sizes['number']):
                    member_data = year_tp.isel(number=n).values
                    daily_precip[0, n] = member_data[0]
                    diff_data = np.diff(member_data, axis=0)
                    daily_precip[1:, n] = diff_data

                # 创建日降水数据集
                daily_tp = xr.DataArray(
                    data=daily_precip,
                    dims=['time', 'number', 'latitude', 'longitude'],
                    coords={
                        'time': adjusted_dates,
                        'number': year_tp['number'],
                        'latitude': year_tp['latitude'],
                        'longitude': year_tp['longitude']
                    }
                )

                # 按周分组 (4周，每周7天)
                for week in range(4):
                    start_day = week * 7
                    end_day = start_day + 7

                    # 使用原始日期提取大气变量
                    original_week_dates = original_dates[start_day:end_day]

                    # 使用调整后的日期提取观测数据
                    adjusted_week_dates = adjusted_dates[start_day:end_day]

                    # 提取该周的日降水数据 (所有成员)
                    week_daily = daily_tp.sel(time=adjusted_week_dates)

                    # 计算周平均
                    week_mean_tp = week_daily.mean(dim='time').mean(dim='number')

                    # 获取大气变量周平均（200和500hPa）
                    atmospheric_data_200_500 = self._extract_atmospheric_data_200_500(
                        ds_gh, ds_pf, ds_pf13012, original_week_dates, lat_range, lon_range
                    )

                    # 获取大气变量周平均（850hPa）
                    atmospheric_data_850 = self._extract_atmospheric_data_850(
                        ds_gh, ds_pf, ds_pf13012, original_week_dates, lat_range, lon_range
                    )

                    # 组合所有输入特征
                    input_features = np.concatenate([
                        np.expand_dims(week_mean_tp.values, axis=0),  # 降水 (通道0)
                        atmospheric_data_200_500,                     # 200和500hPa大气变量 (通道1-12)
                        atmospheric_data_850,                         # 850hPa大气变量 (通道13-18)
                        np.expand_dims(self.dem_data.values, axis=0) # 地形高度 (通道19)
                    ], axis=0)

                    # 获取该周对应的观测数据
                    week_cn = []
                    valid_dates = []
                    for target_date in adjusted_week_dates:
                        if target_date in cn_times:
                            cn_target = ds_cn['pre'].sel(
                                time=target_date,
                                latitude=lat_range,
                                longitude=lon_range
                            )
                            week_cn.append(cn_target)
                            valid_dates.append(target_date)

                    # 检查裁剪
                    if len(week_cn) != 7:
                        print(f"  年份 {year} 第 {week+1} 周: 只有 {len(week_cn)} 天有效观测数据，跳过")
                        continue

                    # 计算CN05周平均日降水
                    cn_weekly = sum(week_cn) / 7.0
                    cn_weekly_values = cn_weekly.values

                    # 添加到数据集
                    self.data.append((
                        input_features,
                        cn_weekly_values,
                        adjusted_week_dates[0],
                        week + 1,
                        year
                    ))

                    self.dates.append(adjusted_week_dates[0])
                    self.lead_times.append(week + 1)
                    self.years.append(year)

            # 关闭文件
            ds_tp.close()
            if ds_gh: ds_gh.close()
            if ds_pf: ds_pf.close()
            if ds_pf13012: ds_pf13012.close()

        if not self.data:
            raise ValueError("数据集样本错误")

        # 记录网格信息
        self.grid_size = daily_tp.latitude.size, daily_tp.longitude.size
        self.latitudes = daily_tp.latitude.values
        self.longitudes = daily_tp.longitude.values

        print(f"\n数据集创建完成，包含 {len(self.data)} 个样本")
        print(f"样本年份分布: {pd.Series(self.years).value_counts().sort_index()}")

        # 计算各通道的最大值
        print("计算各通道最大值用于归一化...")
        self.channel_max = np.zeros(20)  # 现在有20个通道

        # 收集所有样本的数据来计算最大值
        all_inputs = np.array([sample[0] for sample in self.data])

        for ch in range(20):
            # 修改：对所有通道都计算最大值，包括降水通道
            channel_data = all_inputs[:, ch, :, :]
            self.channel_max[ch] = np.nanmax(channel_data)
            print(f"通道 {ch} 最大值: {self.channel_max[ch]:.6f}")

        # 确保没有通道的最大值为0
        self.channel_max[self.channel_max == 0] = 1.0

        # 计算极端降水阈值
        print("计算极端降水阈值...")
        threshold_years = range(2001, 2016)
        all_nonoverlap_weekly = []

        print("提取2001-2015年每年的5月1日到10月30日...")
        for year in threshold_years:
            start_date = datetime(year, 5, 3)
            end_date = datetime(year, 10, 31)

            year_data = ds_cn['pre'].sel(
                time=slice(start_date, end_date),
                latitude=lat_range,
                longitude=lon_range
            )

            year_dates = pd.to_datetime(year_data.time.values)
            num_weeks = len(year_dates) // 7
            for week_idx in range(num_weeks):
                start_index = week_idx * 7
                end_index = start_index + 7
                week_dates = year_dates[start_index:end_index]

                week_data = []
                for date in week_dates:
                    cn_target = year_data.sel(time=date)
                    week_data.append(cn_target)

                week_avg = sum(week_data) / 7.0
                all_nonoverlap_weekly.append(week_avg)

        if not all_nonoverlap_weekly:
            raise ValueError("没有有效的非重叠周数据来计算阈值")

        ds_nonoverlap_weekly = xr.concat(all_nonoverlap_weekly, dim='time')
        thresholds = ds_nonoverlap_weekly.quantile(0.9, dim='time', skipna=True)
        self.thresholds = thresholds.values

        # 确保阈值中没有NaN值
        if np.isnan(self.thresholds).any():
            global_threshold = np.nanmean(self.thresholds)
            self.thresholds = np.nan_to_num(self.thresholds, nan=global_threshold)
            print(f"阈值中存在的NaN值，已使用全局平均值({global_threshold:.2f} mm/day)填充")
        else:
            global_threshold = np.mean(self.thresholds)

        print(f"CN05.1阈值计算完成，使用2001-2015年5月1日-10月2日非重叠7天数据 ({len(all_nonoverlap_weekly)}周)")
        print(f"CN05.1阈值范围: {np.nanmin(self.thresholds):.1f}-{np.nanmax(self.thresholds):.1f} mm/day")
        print(f"全局平均阈值: {global_threshold:.1f} mm/day")

        # 新增：计算EC四周各自的阈值
        print("\n计算EC四周各自的阈值...")
        self.ec_thresholds_by_week = {}

        for week in range(1, 5):
            try:
                ec_data_list = []
                ec_data, _ = self.get_training_data_by_lead(week, train_years=range(2001, 2016))
                if ec_data is not None and len(ec_data) > 0:
                    ec_data_list = list(ec_data)

                if len(ec_data_list) == 0:
                    print(f"警告: 提前期 {week} 周没有EC数据")
                    continue

                # 计算每个网格点的90%分位数（阈值）
                ec_threshold = np.zeros(self.grid_size)

                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        grid_values = []
                        for sample_data in ec_data_list:
                            if not np.isnan(sample_data[i, j]):
                                grid_values.append(sample_data[i, j])

                        if len(grid_values) > 0:
                            ec_threshold[i, j] = np.percentile(grid_values, 90)
                        else:
                            ec_threshold[i, j] = np.nan

                # 应用有效掩膜
                ec_threshold = np.where(self.cn_valid_mask, ec_threshold, np.nan)
                self.ec_thresholds_by_week[week] = ec_threshold

                # 计算统计信息
                valid_ec_threshold = ec_threshold[self.cn_valid_mask]
                if len(valid_ec_threshold) > 0:
                    print(f"提前期 {week} 周EC阈值计算完成，范围: {np.nanmin(valid_ec_threshold):.1f}-{np.nanmax(valid_ec_threshold):.1f} mm/day")
                else:
                    print(f"提前期 {week} 周EC阈值计算完成，但无有效数据")

            except Exception as e:
                print(f"计算第{week}周EC阈值时出错: {e}")
                continue

        ds_cn.close()
        ds_dem.close()

        # 保存到缓存
        self._save_to_cache(cache_file)

    def _create_tibet_mask(self, lats, lons):
        """创建青藏高原掩膜"""
        try:
            from shapely.geometry import Point

            reader = shpreader.Reader(DATA_PATHS['tibet_shapefile'])
            tibet_geometries = list(reader.geometries())
            reader.close()

            mask = np.zeros((len(lats), len(lons)), dtype=bool)

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    point = Point(lon, lat)
                    for geometry in tibet_geometries:
                        if geometry.contains(point):
                            mask[i, j] = True
                            break

            print(f"青藏高原掩膜创建完成，有效点数: {np.sum(mask)}")
            return mask

        except Exception as e:
            print(f"创建青藏高原掩膜时出错: {e}")
            print("使用简化的矩形边界作为备用")
            mask = np.zeros((len(lats), len(lons)), dtype=bool)
            tibet_lon_min, tibet_lon_max = 75, 105
            tibet_lat_min, tibet_lat_max = 25, 40

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    if (tibet_lon_min <= lon <= tibet_lon_max and
                        tibet_lat_min <= lat <= tibet_lat_max):
                        mask[i, j] = True

            print(f"使用备用矩形掩膜，有效点数: {np.sum(mask)}")
            return mask

    def _save_to_cache(self, cache_file):
        """保存数据集到缓存文件"""
        cache_data = {
            'data': self.data,
            'dates': self.dates,
            'lead_times': self.lead_times,
            'years': self.years,
            'grid_size': self.grid_size,
            'latitudes': self.latitudes,
            'longitudes': self.longitudes,
            'channel_max': self.channel_max,
            'thresholds': self.thresholds,
            'cn_valid_mask': self.cn_valid_mask,
            'cn_lat': self.cn_lat,
            'cn_lon': self.cn_lon,
            'ec_latitudes': self.ec_latitudes,
            'ec_longitudes': self.ec_longitudes,
            'ec_thresholds_by_week': self.ec_thresholds_by_week
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"数据集已保存到缓存: {cache_file}")

    def _load_from_cache(self, cache_file):
        """从缓存文件加载数据集"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # 恢复所有属性
        self.data = cache_data['data']
        self.dates = cache_data['dates']
        self.lead_times = cache_data['lead_times']
        self.years = cache_data['years']
        self.grid_size = cache_data['grid_size']
        self.latitudes = cache_data['latitudes']
        self.longitudes = cache_data['longitudes']
        self.channel_max = cache_data['channel_max']
        self.thresholds = cache_data['thresholds']
        self.cn_valid_mask = cache_data['cn_valid_mask']
        self.cn_lat = cache_data['cn_lat']
        self.cn_lon = cache_data['cn_lon']
        self.ec_latitudes = cache_data['ec_latitudes']
        self.ec_longitudes = cache_data['ec_longitudes']
        self.ec_thresholds_by_week = cache_data.get('ec_thresholds_by_week', {})

        print(f"数据集加载完成，包含 {len(self.data)} 个样本")
        print(f"样本年份分布: {pd.Series(self.years).value_counts().sort_index()}")
        print(f"CN05.1阈值范围: {np.nanmin(self.thresholds):.1f}-{np.nanmax(self.thresholds):.1f} mm/day")
        print("归一化策略: 所有通道都进行归一化")
        print(f"有效掩膜点数: {np.sum(self.cn_valid_mask)}")

        # 打印EC阈值信息
        if self.ec_thresholds_by_week:
            print("EC阈值信息:")
            for week in range(1, 5):
                if week in self.ec_thresholds_by_week:
                    ec_threshold = self.ec_thresholds_by_week[week]
                    valid_ec_threshold = ec_threshold[self.cn_valid_mask]
                    if len(valid_ec_threshold) > 0:
                        print(f"  提前期 {week} 周: {np.nanmin(valid_ec_threshold):.1f}-{np.nanmax(valid_ec_threshold):.1f} mm/day")

    def _extract_atmospheric_data_200_500(self, ds_gh, ds_pf, ds_pf13012, week_dates, lat_range, lon_range):
        """提取200和500hPa大气变量周平均数据"""
        atmospheric_data = []

        # 提取gh变量 (200和500hPa)
        if ds_gh is not None:
            target_levels = [200, 500]
            for var in ['gh']:
                if var in ds_gh.data_vars:
                    try:
                        var_data = ds_gh[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])

                        for level in target_levels:
                            if level in var_weekly.level.values:
                                level_data = var_weekly.sel(level=level)
                                atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取 {var} 数据时出错: {e}")

        # 提取w变量 (200和500hPa)
        if ds_pf is not None:
            target_levels = [200, 500]
            for var in ['w']:
                if var in ds_pf.data_vars:
                    try:
                        var_data = ds_pf[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])

                        for level in target_levels:
                            if level in var_weekly.level.values:
                                level_data = var_weekly.sel(level=level)
                                atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取 {var} 数据时出错: {e}")

        # 提取q变量 (200和500hPa)
        if ds_pf is not None:
            target_levels = [200, 500]
            for var in ['q']:
                if var in ds_pf.data_vars:
                    try:
                        var_data = ds_pf[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])

                        for level in target_levels:
                            if level in var_weekly.level.values:
                                level_data = var_weekly.sel(level=level)
                                atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取 {var} 数据时出错: {e}")

        # 提取t, u, v变量 (200和500hPa)
        if ds_pf13012 is not None:
            target_levels = [200, 500]
            for var in ['t','u','v']:
                if var in ds_pf13012.data_vars:
                    try:
                        var_data = ds_pf13012[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])

                        for level in target_levels:
                            if level in var_weekly.level.values:
                                level_data = var_weekly.sel(level=level)
                                atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取 {var} 数据时出错: {e}")

        return np.stack(atmospheric_data, axis=0)

    def _extract_atmospheric_data_850(self, ds_gh, ds_pf, ds_pf13012, week_dates, lat_range, lon_range):
        """提取850hPa大气变量周平均数据"""
        atmospheric_data = []
        target_level = 850

        # 提取850hPa的gh变量
        if ds_gh is not None:
            for var in ['gh']:
                if var in ds_gh.data_vars:
                    try:
                        var_data = ds_gh[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])
                        if target_level in var_weekly.level.values:
                            level_data = var_weekly.sel(level=target_level)
                            atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取850hPa {var} 数据时出错: {e}")

        # 提取850hPa的w变量
        if ds_pf is not None:
            for var in ['w']:
                if var in ds_pf.data_vars:
                    try:
                        var_data = ds_pf[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])
                        if target_level in var_weekly.level.values:
                            level_data = var_weekly.sel(level=target_level)
                            atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取850hPa {var} 数据时出错: {e}")

        # 提取850hPa的q变量
        if ds_pf is not None:
            for var in ['q']:
                if var in ds_pf.data_vars:
                    try:
                        var_data = ds_pf[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])
                        if target_level in var_weekly.level.values:
                            level_data = var_weekly.sel(level=target_level)
                            atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取850hPa {var} 数据时出错: {e}")

        # 提取850hPa的t, u, v变量
        if ds_pf13012 is not None:
            for var in ['t','u','v']:
                if var in ds_pf13012.data_vars:
                    try:
                        var_data = ds_pf13012[var].sel(
                            time=week_dates,
                            latitude=lat_range,
                            longitude=lon_range
                        )
                        var_weekly = var_data.mean(dim=['time', 'number'])
                        if target_level in var_weekly.level.values:
                            level_data = var_weekly.sel(level=target_level)
                            atmospheric_data.append(level_data.values)
                    except Exception as e:
                        print(f"提取850hPa {var} 数据时出错: {e}")

        return np.stack(atmospheric_data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, output, _, lead_time, _ = self.data[idx]

        # 创建输出掩膜
        output_mask = np.isnan(output).astype(np.float32)

        # 确保没有NaN值
        inputs = np.nan_to_num(inputs, nan=0.0)
        output = np.nan_to_num(output, nan=0.0)

        # 修改：对所有通道进行归一化，包括降水通道
        normalized_inputs = inputs.copy()
        for ch in range(20):  # 现在有20个通道
            if self.channel_max[ch] > 0:
                normalized_inputs[ch] = inputs[ch] / self.channel_max[ch]

        return (
            torch.tensor(normalized_inputs, dtype=torch.float32),
            torch.tensor(output, dtype=torch.float32),
            torch.tensor(output_mask, dtype=torch.float32),
            lead_time
        )

    def get_grid_info(self):
        """返回网格信息"""
        return {
            'latitudes': self.latitudes,
            'longitudes': self.longitudes,
            'grid_size': self.grid_size
        }

    def get_thresholds(self, lead_time=None):
        """返回阈值数组"""
        if lead_time is not None and lead_time in self.ec_thresholds_by_week:
            return self.ec_thresholds_by_week[lead_time]
        else:
            return self.thresholds

    def get_ec_thresholds_by_week(self, lead_time):
        """返回指定提前期的EC阈值"""
        return self.ec_thresholds_by_week.get(lead_time, None)

    def get_cn_valid_mask(self):
        """返回CN05.1有效掩膜"""
        return self.cn_valid_mask

    def get_training_data_by_lead(self, lead_time, train_years):
        """获取指定提前期的训练数据"""
        ec_data = []
        obs_data = []

        for idx, (ec, obs, _, lt, year) in enumerate(self.data):
            if lt == lead_time and year in train_years:
                ec_precip = ec[0]  # 第0通道是降水
                ec_data.append(ec_precip)
                obs_data.append(obs)

        return np.array(ec_data), np.array(obs_data)

    def get_data_statistics(self):
        """返回数据统计信息"""
        sample_input, sample_target, _, _ = self[0]

        stats = {
            'input_shape': sample_input.shape,
            'target_shape': sample_target.shape,
            'channel_stats': {}
        }

        # 通道说明
        channel_names = [
            'precipitation',
            'gh_200hpa', 'gh_500hpa',
            'w_200hpa', 'w_500hpa',
            'q_200hpa', 'q_500hpa',
            't_200hpa', 't_500hpa',
            'u_200hpa', 'u_500hpa',
            'v_200hpa', 'v_500hpa',
            'gh_850hpa', 'w_850hpa', 'q_850hpa',
            't_850hpa', 'u_850hpa', 'v_850hpa',
            'dem'
        ]

        for ch in range(20):
            channel_data = sample_input[ch]
            stats['channel_stats'][ch] = {
                'name': channel_names[ch],
                'min': channel_data.min().item(),
                'max': channel_data.max().item(),
                'mean': channel_data.mean().item(),
                'normalized': True,
                'normalization_factor': self.channel_max[ch]
            }

        stats['target_stats'] = {
            'min': sample_target.min().item(),
            'max': sample_target.max().item(),
            'mean': sample_target.mean().item()
        }

        return stats


def compute_obs_threshold_period(years, lat_range=LAT_RANGE, lon_range=LON_RANGE,
                                 percentile=0.9):
    """按给定年份区间，用 CN05.1 观测逐日降水计算极端降水阈值的空间场（mm/day）。

    与 WeeklyEC_S2S_Dataset.__init__ 中的阈值计算逻辑完全一致：
    每年取 5月3日—10月31日，按非重叠 7 天求周平均，把所有年份的周序列汇总后，
    再对全体周求 ``percentile`` 分位数，得到每个网格点的阈值。

    参数
    ----
    years : iterable of int
        用于计算阈值的年份（例如 range(2001, 2016) 表示 2001-2015）。
    lat_range, lon_range : slice
        区域选择，默认取 config.LAT_RANGE/LON_RANGE。
    percentile : float
        分位数，默认 0.9（90 分位）。

    返回
    ----
    (thresholds, lat, lon) : (np.ndarray, np.ndarray, np.ndarray)
        thresholds 为每个格点的阈值场（mm/day，与 lat/lon 同维），
        lat/lon 为对应网格坐标。
    """
    import xarray as xr
    from datetime import datetime

    ds_cn = xr.open_dataset(DATA_PATHS['cn05_file'])
    ds_cn = ds_cn.sel(latitude=lat_range, longitude=lon_range)

    all_nonoverlap_weekly = []
    for year in sorted(years):
        start_date = datetime(year, 5, 3)
        end_date = datetime(year, 10, 31)

        year_data = ds_cn['pre'].sel(
            time=slice(start_date, end_date)
        )
        year_dates = pd.to_datetime(year_data.time.values)
        num_weeks = len(year_dates) // 7

        for week_idx in range(num_weeks):
            start_index = week_idx * 7
            end_index = start_index + 7
            week_dates = year_dates[start_index:end_index]

            week_data = []
            for date in week_dates:
                cn_target = year_data.sel(time=date)
                week_data.append(cn_target)

            week_avg = sum(week_data) / 7.0
            all_nonoverlap_weekly.append(week_avg)

    if not all_nonoverlap_weekly:
        ds_cn.close()
        raise ValueError("没有有效的非重叠周数据来计算阈值")

    ds_nonoverlap_weekly = xr.concat(all_nonoverlap_weekly, dim='time')
    thresholds = ds_nonoverlap_weekly.quantile(percentile, dim='time', skipna=True)
    th = np.asarray(thresholds.values, dtype=float)

    # NaN 处理：与 __init__ 保持一致，用全局平均填充
    if np.isnan(th).any():
        global_threshold = np.nanmean(th)
        th = np.nan_to_num(th, nan=global_threshold)

    lat = np.asarray(ds_cn.latitude.values, dtype=float)
    lon = np.asarray(ds_cn.longitude.values, dtype=float)
    ds_cn.close()

    print(f"观测阈值计算完成：{sorted(years)[0]}-{sorted(years)[-1]}，"
          f"范围 {np.nanmin(th):.2f}-{np.nanmax(th):.2f} mm/day，"
          f"均值 {np.nanmean(th):.2f} mm/day")
    return th, lat, lon
