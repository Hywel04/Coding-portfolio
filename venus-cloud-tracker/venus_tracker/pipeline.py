#Imports
import numpy as np
import pandas as pd
from venus_tracker.tracker import VenusPhysicalTracker

# Pipeline for Venus cloud motion tracking
def execute_tracking_pipeline(results_t1, results_t2, df_t1, df_t2, dt_seconds):
    tracker = VenusPhysicalTracker(deg_per_px=0.125, dt_seconds=dt_seconds)

    lats_t1, lons_t1 = results_t1['latitude'], results_t1['longitude']
    lats_t2, lons_t2 = results_t2['latitude'], results_t2['longitude']
    img1_data, img2_data = results_t1['corrected_rad'], results_t2['corrected_rad']

    feat_candidates = {}
    for _, row in df_t1.iterrows():
        # Decoupled grid mapping
        r_t1 = np.abs(lats_t1 - row['centroid_lat']).argmin()
        c_t1 = np.abs(lons_t1 - row['centroid_lon']).argmin()

        r_t2 = np.abs(lats_t2 - row['centroid_lat']).argmin()
        c_t2 = np.abs(lons_t2 - row['centroid_lon']).argmin()

        cands = tracker.get_ccs_candidates(
            img1_data, img2_data,
            r_t1, c_t1, r_t2, c_t2,
            lat=row['centroid_lat']
        )
        if cands:
            feat_candidates[row['feature_id']] = cands

    distances = {}
    ids = list(feat_candidates.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            f1, f2 = df_t1[df_t1['feature_id'] == ids[i]].iloc[0], df_t1[df_t1['feature_id'] == ids[j]].iloc[0]
            dist = np.sqrt((f1['centroid_lat']-f2['centroid_lat'])**2 + (f1['centroid_lon']-f2['centroid_lon'])**2)
            distances[(ids[i], ids[j])] = distances[(ids[j], ids[i])] = dist

    final_matches = tracker.run_relaxation(feat_candidates, distances)

    raw_results = []
    for f_id, winner in final_matches.items():
        if winner is None: continue
        feature_origin = df_t1[df_t1['feature_id'] == f_id].iloc[0]
        raw_results.append({
            'feature_id': f_id,
            'u': float(winner['vec'][0]), 'v': float(winner['vec'][1]),
            'u_ms': winner['v_ms'][0].item(), 'v_ms': winner['v_ms'][1].item(),
            'centroid_lat': feature_origin['centroid_lat'],
            'centroid_lon': feature_origin['centroid_lon']
        })

    if not raw_results: return pd.DataFrame()

    # Flow Clustering Post-Process
    n = len(raw_results)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((raw_results[i]['centroid_lat'] - raw_results[j]['centroid_lat'])**2 +
                           (raw_results[i]['centroid_lon'] - raw_results[j]['centroid_lon'])**2)
            if dist > 15.0: continue

            vec_i = np.array([raw_results[i]['u'], raw_results[i]['v']])
            vec_j = np.array([raw_results[j]['u'], raw_results[j]['v']])
            diff_sq = np.sum((vec_i - vec_j)**2)
            L_kl_sq = (tracker.alpha * dist)**2
            c_kl = np.exp(-np.log(2) * diff_sq / (L_kl_sq + 1e-6))

            if c_kl > 0.5:
                adj[i, j] = adj[j, i] = True

    visited = np.zeros(n, dtype=bool)
    largest_group = []
    isolated_nodes = [i for i in range(n) if not np.any(adj[i])]

    for i in range(n):
        if not visited[i] and i not in isolated_nodes:
            group = []
            q = [i]
            visited[i] = True
            while q:
                curr = q.pop(0)
                group.append(curr)
                for neighbor in range(n):
                    if adj[curr, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            if len(group) > len(largest_group):
                largest_group = group

    final_keep_indices = list(set(largest_group + isolated_nodes))
    if not final_keep_indices: return pd.DataFrame()

    return pd.DataFrame([raw_results[idx] for idx in final_keep_indices]).drop(columns=['centroid_lat', 'centroid_lon'])

def get_evolution_deltas(df_evo, df_t2, next_global_id):
    """Calculates morphologic changes with Area Consistency checks."""

    columns = ['feature_id', 'u_ms', 'v_ms', 'area_delta_pct', 'contrast_delta',
               'centroid_lat', 'centroid_lon', 'u', 'v', 'time_t1', 'time_t2', 'solar_angle_t1']

    df_t2_updated = df_t2.copy()
    df_t2_updated['is_matched'] = False
    final_stats = []

    for _, f1 in df_evo.iterrows():
        pred_lat, pred_lon = f1['centroid_lat'] + f1['v'], f1['centroid_lon'] + f1['u']

        df_t2_updated['dist'] = np.sqrt((df_t2_updated['centroid_lat'] - pred_lat)**2 +
                                        (df_t2_updated['centroid_lon'] - pred_lon)**2)

        # Calculate how much the candidate T2 clouds have grown/shrunk compared to T1
        df_t2_updated['area_ratio'] = df_t2_updated['area_km2'] / f1['area_km2']

        # Apply physical limits to area growth/collapse
        # Must be within 4.0 degrees, AND cannot shrink below 20% or grow above 300% (Merging)
        valid_matches = df_t2_updated[(df_t2_updated['dist'] < 4.0) &
                                      (df_t2_updated['area_ratio'] >= 0.20) &
                                      (df_t2_updated['area_ratio'] <= 3.0)].sort_values('dist')

        if not valid_matches.empty:
            match_idx = valid_matches.index[0]
            f2 = valid_matches.iloc[0]

            df_t2_updated.at[match_idx, 'feature_id'] = f1['feature_id']
            df_t2_updated.at[match_idx, 'is_matched'] = True

            final_stats.append({
                'feature_id': f1['feature_id'],
                'u_ms': f1['u_ms'], 'v_ms': f1['v_ms'],
                'area_delta_pct': ((f2['area_km2'] - f1['area_km2']) / f1['area_km2']) * 100,
                'contrast_delta': f2['mean_contrast'] - f1['mean_contrast'],
                'centroid_lat': f1['centroid_lat'], 'centroid_lon': f1['centroid_lon'],
                'u': f1['u'], 'v': f1['v'],
                'solar_angle_t1': f1['solar_angle_t1']
            })

    for idx, row in df_t2_updated.iterrows():
        if not row['is_matched']:
            df_t2_updated.at[idx, 'feature_id'] = next_global_id
            next_global_id += 1

    df_t2_updated.drop(columns=['dist', 'area_ratio', 'is_matched'], inplace=True, errors='ignore')

    return pd.DataFrame(final_stats) if final_stats else pd.DataFrame(columns=columns), df_t2_updated, next_global_id

