"""Reusable diagnostics for South Sudan hydrometeorological analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pyarrow.dataset as arrow_dataset
import rasterio
from rasterio.mask import mask as raster_mask
from rasterio.transform import rowcol


AWEIL_COUNTIES = (
    'Aweil Centre', 'Aweil East', 'Aweil North', 'Aweil South', 'Aweil West'
)
FLOOD_PIXEL_STEP = 10 / 4800
FLOOD_PIXEL_RADIUS_M = 6371007.181


def era5_daily_table(dataset: xr.Dataset, bbox: dict[str, float] | None = None) -> pd.DataFrame:
    """Spatially average ERA5 daily precipitation and runoff in a tidy table."""
    selected = dataset[['tp', 'ro']]
    if bbox is not None:
        selected = selected.sel(
            latitude=slice(bbox['lat_max'], bbox['lat_min']),
            longitude=slice(bbox['lon_min'], bbox['lon_max']),
        )
    table = selected.mean(dim=('latitude', 'longitude'), skipna=True).to_dataframe()
    table.index = pd.to_datetime(table.index)
    table = table.rename_axis('date').reset_index()
    table[['tp', 'ro']] = table[['tp', 'ro']] * 1000.0
    return table.rename(columns={'tp': 'precipitation_mm', 'ro': 'runoff_mm'})


def et_daily_table(et_data: dict[int, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    """Normalize processed ET CSV data to ``date`` and ``et_mm`` columns."""
    if isinstance(et_data, dict):
        frames = []
        for frame in et_data.values():
            current = frame.copy()
            if 'date' in current.columns:
                current = current.set_index('date')
            current.index = pd.to_datetime(current.index)
            frames.append(current)
        if not frames:
            return pd.DataFrame(columns=['date', 'et_mm'])
        et_data = pd.concat(frames)
    frame = et_data.copy()
    if 'date' in frame.columns:
        frame = frame.set_index('date')
    frame.index = pd.to_datetime(frame.index)
    numeric_columns = frame.select_dtypes('number').columns
    value_column = 'gridcell' if 'gridcell' in frame.columns else numeric_columns[0]
    result = frame[[value_column]].rename(columns={value_column: 'et_mm'})
    return result.rename_axis('date').reset_index().drop_duplicates('date').sort_values('date')


def water_budget(daily_hydro: pd.DataFrame, daily_et: pd.DataFrame | None = None,
                 interval: str = 'M') -> pd.DataFrame:
    """Calculate interval P - ET0 and the runoff coefficient."""
    required = {'date', 'precipitation_mm', 'runoff_mm'}
    missing = required - set(daily_hydro.columns)
    if missing:
        raise ValueError(f'Missing hydrological columns: {sorted(missing)}')
    table = daily_hydro.copy()
    table['date'] = pd.to_datetime(table['date'])
    if daily_et is not None:
        table = table.merge(et_daily_table(daily_et), on='date', how='left')
    else:
        table['et_mm'] = np.nan
    table['period'] = table['date'].dt.to_period(interval).dt.to_timestamp()
    result = table.groupby('period', as_index=False).agg(
        precipitation_mm=('precipitation_mm', 'sum'),
        runoff_mm=('runoff_mm', 'sum'),
        et_mm=('et_mm', lambda values: values.sum(min_count=1)),
        et_days=('et_mm', lambda values: values.notna().sum()),
    )
    result['net_water_balance_mm'] = result['precipitation_mm'] - result['et_mm']
    result['runoff_coefficient'] = result['runoff_mm'].div(
        result['precipitation_mm'].replace(0, np.nan)
    )
    result['water_balance_class'] = np.select(
        [result['net_water_balance_mm'] > 0, result['net_water_balance_mm'] < 0],
        ['surplus', 'deficit'], default='balanced'
    )
    return result


def monthly_climatology(daily: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Return monthly climatological means and standard deviations."""
    table = daily.copy()
    table['date'] = pd.to_datetime(table['date'])
    return table.assign(month=table['date'].dt.month).groupby('month')[value_columns].agg(
        ['mean', 'std']
    ).reset_index()


def percentile_extremes(daily: pd.DataFrame, value_column: str,
                        percentiles: tuple[float, ...] = (0.95, 0.99)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count threshold exceedances and consecutive extreme runs by year."""
    table = daily[['date', value_column]].copy()
    table['date'] = pd.to_datetime(table['date'])
    table[value_column] = pd.to_numeric(table[value_column], errors='coerce')
    thresholds = {p: table[value_column].quantile(p) for p in percentiles}
    rows = []
    for year, group in table.dropna().groupby(table['date'].dt.year):
        row = {'year': int(year), 'n_observations': len(group)}
        for percentile, threshold in thresholds.items():
            extreme = group[value_column] >= threshold
            starts = extreme & ~extreme.shift(fill_value=False)
            key = f'p{int(percentile * 100)}'
            row[f'{key}_threshold'] = threshold
            row[f'{key}_days'] = int(extreme.sum())
            row[f'{key}_events'] = int(starts.sum())
            row[f'{key}_max_duration_days'] = int(
                extreme.groupby((~extreme).cumsum()).sum().max() if extreme.any() else 0
            )
        rows.append(row)
    threshold_table = pd.DataFrame([{'percentile': p, 'threshold': value} for p, value in thresholds.items()])
    return threshold_table, pd.DataFrame(rows)


def lagged_correlation(first: pd.Series, second: pd.Series, max_lag: int = 180) -> pd.DataFrame:
    """Correlate two daily series while shifting the second series by each lag."""
    left = pd.Series(first, dtype='float64').rename('first')
    right = pd.Series(second, dtype='float64').rename('second')
    frame = pd.concat([left, right], axis=1).sort_index()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        aligned = frame.assign(second=frame['second'].shift(lag)).dropna()
        rows.append({'lag_days': lag, 'correlation': aligned['first'].corr(aligned['second']),
                     'n_pairs': len(aligned)})
    return pd.DataFrame(rows)


def flood_event_metrics(flood_masks: pd.DataFrame) -> pd.DataFrame:
    """Summarize flood pixels, unusual share, and event duration by year and type."""
    columns = ['year', 'flood_type', 'flood_pixels', 'flood_dates', 'max_duration_days']
    if flood_masks.empty:
        return pd.DataFrame(columns=columns)
    table = flood_masks.copy()
    table['date'] = pd.to_datetime(table['date']).dt.normalize()
    table = table.drop_duplicates(['date', 'lat', 'lon', 'flood_type'])
    rows = []
    for (year, flood_type), group in table.groupby([table['date'].dt.year, 'flood_type']):
        dates = pd.Series(group['date'].drop_duplicates().sort_values())
        runs = dates.diff().dt.days.ne(1).cumsum()
        rows.append({'year': int(year), 'flood_type': int(flood_type),
                     'flood_pixels': len(group), 'flood_dates': dates.nunique(),
                     'max_duration_days': int(dates.groupby(runs).size().max())})
    return pd.DataFrame(rows).sort_values(['year', 'flood_type']).reset_index(drop=True)


def quality_report(series: pd.Series, name: str, expected_start=None, expected_end=None) -> pd.DataFrame:
    """Report duplicate timestamps, missing values, invalid negatives, and coverage."""
    values = pd.to_numeric(series, errors='coerce')
    dates = pd.DatetimeIndex(series.index)
    report = {
        'name': name, 'n_rows': len(series), 'n_unique_dates': dates.normalize().nunique(),
        'duplicate_dates': int(dates.normalize().duplicated().sum()),
        'missing_values': int(values.isna().sum()), 'negative_values': int((values < 0).sum()),
        'zero_values': int((values == 0).sum()), 'start': dates.min(), 'end': dates.max(),
    }
    if expected_start is not None and expected_end is not None:
        expected = pd.date_range(expected_start, expected_end, freq='D')
        report['missing_expected_dates'] = int(expected.difference(dates.normalize()).size)
    return pd.DataFrame([report])


def lake_altimetry_quality(lake_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flag invalid altimetry readings and return baseline-relative lake levels."""
    rows = []
    for lake, frame in lake_data.items():
        table = frame.copy()
        value_column = 'water_level' if 'water_level' in table else 'height_wrt_ref'
        values = pd.to_numeric(table[value_column], errors='coerce')
        invalid_flags = pd.Series(False, index=table.index)
        if 'mode1' in table:
            # Hydroweb mode 1 codes 3 and 4 are valid measurement modes.
            invalid_flags |= ~pd.to_numeric(table['mode1'], errors='coerce').isin([3, 4])
        for flag in ('ice_flag', 'data_source_flag'):
            if flag in table:
                invalid_flags |= pd.to_numeric(table[flag], errors='coerce').fillna(1).ne(0)
        valid = values.notna() & values.gt(0) & ~invalid_flags
        anomaly = values - values[valid].mean()
        rows.append(pd.DataFrame({
            'lake': lake, 'date': pd.to_datetime(table.index), 'water_level': values,
            'valid': valid, 'invalid_flag': invalid_flags, 'anomaly': anomaly,
        }))
    if not rows:
        return pd.DataFrame(columns=['lake', 'date', 'water_level', 'valid', 'invalid_flag', 'anomaly'])
    return pd.concat(rows, ignore_index=True).sort_values(['lake', 'date'])


def era5_spatial_summary(dataset: xr.Dataset) -> pd.DataFrame:
    """Return annual mean precipitation and runoff for every ERA5 grid cell in mm."""
    annual = dataset[['tp', 'ro']].resample(valid_time='YS').mean() * 1000.0
    return annual.to_dataframe().reset_index().rename(columns={
        'valid_time': 'year', 'tp': 'mean_precipitation_mm_per_day',
        'ro': 'mean_runoff_mm_per_day',
    })


def load_aweil_counties(boundaries_path) -> gpd.GeoDataFrame:
    """Load the five Northern Bahr el Ghazal Aweil county polygons."""
    counties = gpd.read_file(boundaries_path).to_crs('EPSG:4326')
    selected = counties[counties['adm2_name'].isin(AWEIL_COUNTIES)].copy()
    if len(selected) != len(AWEIL_COUNTIES):
        raise ValueError('The boundary file does not contain all five Aweil counties')
    return selected[['adm2_name', 'geometry']].rename(columns={'adm2_name': 'county'})


def load_aweil_flood_observations(flood_root, counties: gpd.GeoDataFrame,
                                  years=(2024,)) -> pd.DataFrame:
    """Load recurring and unusual flood pixels and assign each to an Aweil county."""
    bounds = tuple(counties.total_bounds)
    location_filter = (
        (arrow_dataset.field('lon') >= bounds[0]) &
        (arrow_dataset.field('lon') <= bounds[2]) &
        (arrow_dataset.field('lat') >= bounds[1]) &
        (arrow_dataset.field('lat') <= bounds[3])
    )
    parts = []
    for year in years:
        for flood_type in ('recurring', 'unusual'):
            path = flood_root / f'compact_{flood_type}' / f'flood_events_h20v08_{year}.parquet'
            frame = arrow_dataset.dataset(path, format='parquet').to_table(
                columns=['date', 'lat', 'lon', 'cloud_frac'], filter=location_filter
            ).to_pandas()
            frame['flood_type'] = flood_type
            parts.append(frame)
    observations = pd.concat(parts, ignore_index=True)
    observations['date'] = pd.to_datetime(observations['date']).dt.normalize()
    points = gpd.GeoDataFrame(
        observations, geometry=gpd.points_from_xy(observations['lon'], observations['lat']),
        crs='EPSG:4326'
    )
    located = gpd.sjoin(points, counties, how='inner', predicate='within')
    return pd.DataFrame(located.drop(columns=['geometry', 'index_right']))


def daily_flood_pixels(observations: pd.DataFrame) -> pd.DataFrame:
    """Count unique active flood pixels per day and county."""
    data = observations.drop_duplicates(['date', 'lat', 'lon', 'county']).copy()
    return data.groupby(['date', 'county'], as_index=False).agg(
        active_flood_pixels=('lat', 'size'),
        unusual_flood_pixels=('flood_type', lambda values: (values == 'unusual').sum()),
        mean_cloud_fraction=('cloud_frac', 'mean'),
    )


def lag_feature_correlations(signals: pd.DataFrame, flood_target: pd.Series,
                             lags=range(0, 15)) -> pd.DataFrame:
    """Correlate signals from t-k with active flood pixels at t."""
    target = pd.Series(flood_target, dtype='float64').rename('flood_pixels')
    frame = signals.join(target, how='inner').sort_index()
    rows = []
    for lag in lags:
        for signal in signals.columns:
            aligned = pd.concat([frame[signal].shift(lag), frame['flood_pixels']], axis=1).dropna()
            rows.append({
                'lag_days': lag, 'signal': signal,
                'correlation': aligned.iloc[:, 0].corr(aligned['flood_pixels']),
                'n_days': len(aligned),
            })
    return pd.DataFrame(rows)


def rolling_moisture_features(daily: pd.DataFrame, et: pd.DataFrame | None = None,
                              windows=(3, 7, 14, 30)) -> pd.DataFrame:
    """Create antecedent rainfall-minus-ET and runoff-coefficient features."""
    table = daily.set_index('date').sort_index().copy()
    if et is not None:
        table = table.join(et_daily_table(et).set_index('date')['et_mm'])
    else:
        table['et_mm'] = np.nan
    for window in windows:
        table[f'net_moisture_{window}d_mm'] = (
            table['precipitation_mm'].rolling(window).sum() - table['et_mm'].rolling(window).sum()
        )
    table['runoff_share'] = table['runoff_mm'].div(table['precipitation_mm'].replace(0, np.nan))
    return table.reset_index()


def flood_pixel_area_ha(latitude) -> np.ndarray:
    """Return the approximate area of a MODIS flood pixel in hectares."""
    latitude = np.asarray(latitude, dtype=float)
    half_step = FLOOD_PIXEL_STEP / 2
    south = np.deg2rad(latitude - half_step)
    north = np.deg2rad(latitude + half_step)
    area_m2 = FLOOD_PIXEL_RADIUS_M**2 * np.deg2rad(FLOOD_PIXEL_STEP) * (np.sin(north) - np.sin(south))
    return area_m2 / 10000


def agricultural_baseline(counties: gpd.GeoDataFrame, raster_paths: dict[str, str]) -> pd.DataFrame:
    """Calculate available crop and rangeland hectares in each Aweil county."""
    rows = []
    for _, county in counties.iterrows():
        row = {'county': county['county']}
        for land_type, path in raster_paths.items():
            with rasterio.open(path) as source:
                values, window_transform = raster_mask(
                    source, [county.geometry], crop=True, filled=False
                )
                values = values[0]
                row_numbers = np.arange(values.shape[0])
                latitudes = window_transform.f + (row_numbers + 0.5) * window_transform.e
                half_lat = abs(window_transform.e) / 2
                row_area_ha = (
                    FLOOD_PIXEL_RADIUS_M**2 * np.deg2rad(abs(window_transform.a))
                    * (np.sin(np.deg2rad(latitudes + half_lat))
                       - np.sin(np.deg2rad(latitudes - half_lat))) / 10000
                )
                weighted = values.astype(float).filled(np.nan) / 100 * row_area_ha[:, None]
                row[f'{land_type}_hectares'] = float(np.nansum(weighted))
        rows.append(row)
    return pd.DataFrame(rows)


def seasonal_agricultural_exposure(observations: pd.DataFrame, counties: gpd.GeoDataFrame,
                                   raster_paths: dict[str, str]) -> pd.DataFrame:
    """Estimate weekly detected crop/rangeland exposure by county and crop phase.

    Exposure is a detected MODIS-pixel proxy: it is not a polygon inundation area.
    """
    data = observations.drop_duplicates(['date', 'lat', 'lon', 'county', 'flood_type']).copy()
    data['phase'] = pd.cut(
        data['date'].dt.month, bins=[0, 4, 6, 9, 12],
        labels=['dry season', 'planting and germination', 'vegetative growth and flowering', 'maturation and harvest'],
    )
    data['month'] = data['date'].dt.month
    with rasterio.open(next(iter(raster_paths.values()))) as source:
        rows, columns = rowcol(
            source.transform, data['lon'].to_numpy(), data['lat'].to_numpy()
        )
    data['_mask_cell'] = list(zip(rows, columns))
    # Multiple 232 m MODIS detections can fall in one 500 m agricultural cell.
    data = data.drop_duplicates(['county', 'phase', 'month', 'flood_type', '_mask_cell'])
    for land_type, path in raster_paths.items():
        with rasterio.open(path) as source:
            samples = list(source.sample(zip(data['lon'], data['lat'])))
        data[f'{land_type}_fraction'] = np.asarray(samples, dtype=float).ravel()
        with rasterio.open(path) as source:
            pixel_ha = abs(source.transform.a) * abs(source.transform.e) * 111_320**2 / 10000
        data[f'{land_type}_exposed_ha'] = pixel_ha * data[f'{land_type}_fraction'] / 100
    return data.groupby(['county', 'phase', 'month', 'flood_type'], observed=True).agg(
        detected_pixels=('lat', 'size'),
        crop_exposed_hectares=('crop_exposed_ha', 'sum'),
        rangeland_exposed_hectares=('rangeland_exposed_ha', 'sum'),
    ).reset_index()