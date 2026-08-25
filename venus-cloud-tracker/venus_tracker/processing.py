#Imports
import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, median_filter, binary_dilation, label
from skimage.measure import regionprops

#Common processing functions for Venus cloud motion tracking
def get_auto_bounds(ds, buffer_deg=6.0):
    """Scans the solar incidence angle to find the valid dayside disk, padded outward. Automatically sets bounds on data"""
    inangle_2d = ds['inangle'].squeeze().values
    lats = ds.latitude.values
    lons = ds.longitude.values

    cos_i = np.cos(np.radians(inangle_2d))
    valid_mask = cos_i > 0.087

    if not np.any(valid_mask):
        return None

    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    valid_lat_indices = np.where(valid_rows)[0]
    valid_lon_indices = np.where(valid_cols)[0]

    lat_edges = [lats[valid_lat_indices.min()], lats[valid_lat_indices.max()]]
    lon_edges = [lons[valid_lon_indices.min()], lons[valid_lon_indices.max()]]
    return {
        'lat_min': max(-90.0, min(lat_edges) + buffer_deg),
        'lat_max': min(90.0, max(lat_edges) - buffer_deg),
        'lon_min': min(lon_edges) + buffer_deg,
        'lon_max': max(lon_edges) - buffer_deg
    }

def final_median(ds, lat_min, lat_max, lon_min, lon_max, k_size=300, scale_factor=0.25):
    """Processes radiance into cloud features."""
    subset = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    rad_2d = subset['radiance'].squeeze().values
    inangle_2d = subset['inangle'].squeeze().values

    # 1. Base Photometric Mask
    cos_i = np.cos(np.radians(inangle_2d))

    # Cut into disk to ignore edges
    base_valid_mask = cos_i > 0.25

    cos_i_safe = np.where(base_valid_mask, cos_i, np.nan)
    f_corrected = rad_2d / cos_i_safe

    valid_mask = ~np.isnan(f_corrected)
    if not np.any(valid_mask): return None

    # 3. Background subtraction
    img_filled = np.copy(f_corrected)
    img_filled[~valid_mask] = np.nanmedian(f_corrected)

    small_img = zoom(img_filled, scale_factor, order=1)
    k_small = int(k_size * scale_factor)
    if k_small % 2 == 0: k_small += 1
    small_bg = median_filter(small_img, size=k_small)
    background = zoom(small_bg, (rad_2d.shape[0]/small_bg.shape[0], rad_2d.shape[1]/small_bg.shape[1]), order=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        contrast = (f_corrected - background) / background

    # 4. Filter Features
    feature_mask = (contrast <= -0.15) & (contrast >= -0.30)
    detected_features = np.full_like(f_corrected, np.nan)
    detected_features[feature_mask] = contrast[feature_mask]

    detected_features[~valid_mask] = np.nan

    return {
        "latitude": subset.latitude.values, "longitude": subset.longitude.values,
        "contrast": contrast, "features": detected_features,
        "background": background, "corrected_rad": f_corrected,
        "inangle": inangle_2d
    }

def calculate_feature_metrics(results, r_venus=6051.8, min_pixel_count=5, frame_time=None, raw_save_path="raw_features_master.csv"):
    """Extracts properties, applying a strict Boundary Touch Veto, and saves raw unbiased data."""
    columns = ["feature_id", "area_km2", "centroid_lat", "centroid_lon", "mean_contrast", "pixel_count", "solar_angle_t1"]
    if results is None: return pd.DataFrame(columns=columns)

    mask = ~np.isnan(results['features'])
    lats_deg, lons_deg = results['latitude'], results['longitude']
    inangle_2d = results['inangle']

    # Dealing with the Terminator
    # 1. Recreate our 75-degree geometry mask
    valid_mask = np.cos(np.radians(inangle_2d)) > 0.25

    # 2. Define the invalid data
    invalid_zone = ~valid_mask

    # 3. Add the absolute outer edges of the 2D array to the Void just in case
    invalid_zone[0, :] = True
    invalid_zone[-1, :] = True
    invalid_zone[:, 0] = True
    invalid_zone[:, -1] = True

    # 4. Dilate the Void by 15 pixels to create a forbidden area
    forbidden_edge = binary_dilation(invalid_zone, iterations=15)

    # Spherical Area Calculation
    dlat = np.abs(np.mean(np.diff(np.radians(lats_deg))))
    dlon = np.abs(np.mean(np.diff(np.radians(lons_deg))))
    lat_mesh, _ = np.meshgrid(np.radians(lats_deg), np.radians(lons_deg), indexing='ij')
    pixel_area_map = (r_venus**2) * np.cos(lat_mesh) * dlat * dlon

    labeled_array, _ = label(mask.astype(int))
    regions = regionprops(labeled_array, intensity_image=results['contrast'])

    feature_data = []
    raw_feature_data = []

    for current_id, region in enumerate(regions, start=1):

        # Grabs the feature before it can be deleted by the tracking vetoes
        idx_row_raw = np.clip(int(region.centroid[0]), 0, len(lats_deg) - 1)
        idx_col_raw = np.clip(int(region.centroid[1]), 0, len(lons_deg) - 1)

        raw_feature_data.append({
            "timestamp": frame_time,
            "feature_label": current_id,
            "area_pixels": int(region.area),
            "mean_contrast": float(region.mean_intensity),
            "centroid_lat": float(lats_deg[idx_row_raw]),
            "centroid_lon": float(lons_deg[idx_col_raw])
        })


        if region.area < min_pixel_count: continue

        coords = region.coords

        # 1. No go zone
        if np.any(forbidden_edge[coords[:, 0], coords[:, 1]]):
            continue

        # 2.Large scale
        min_row, min_col, max_row, max_col = region.bbox

        lat_spread = abs(lats_deg[min_row] - lats_deg[max_row - 1])
        lon_spread = abs(lons_deg[min_col] - lons_deg[max_col - 1])

        # Latitude limit = 20 degrees (prevents vertical terminator strips)
        if lat_spread > 20.0:
            continue

        total_area = float(np.sum(pixel_area_map[coords[:, 0], coords[:, 1]]))
        idx_row = np.clip(int(region.centroid[0]), 0, len(lats_deg) - 1)
        idx_col = np.clip(int(region.centroid[1]), 0, len(lons_deg) - 1)

        mean_solar_angle = np.nanmean(inangle_2d[coords[:, 0], coords[:, 1]])

        feature_data.append({
            "feature_id": current_id, "area_km2": total_area,
            "centroid_lat": float(lats_deg[idx_row]), "centroid_lon": float(lons_deg[idx_col]),
            "mean_contrast": float(region.mean_intensity), "pixel_count": int(region.area),
            "solar_angle_t1": float(mean_solar_angle)
        })

    # Append raw data to seperate csv
    #if raw_feature_data and raw_save_path:
        #df_raw = pd.DataFrame(raw_feature_data)
        #file_exists = os.path.isfile(raw_save_path)
        #df_raw.to_csv(raw_save_path, mode='a', index=False, header=not file_exists)

    return pd.DataFrame(feature_data) if feature_data else pd.DataFrame(columns=columns)
