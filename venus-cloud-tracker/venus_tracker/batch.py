#Imports
import pandas as pd
from venus_tracker.processing import get_auto_bounds, final_median, calculate_feature_metrics
from venus_tracker.pipeline import execute_tracking_pipeline, get_evolution_deltas
from venus_tracker.visualization import plot_corrected_velocimetry

# Batch processing module for Venus cloud motion tracking
class VenusBatchTracker:
    def __init__(self, dataset_list, output_csv="venus_tracking_log.csv"):
        self.datasets = dataset_list
        self.output_csv = output_csv
        self.master_log = []

    def run_all(self):
        print(f"Starting batch process for {len(self.datasets)} timesteps...")

        idx = 0
        res_prev, df_prev = None, None
        bounds = None

        while res_prev is None and idx < len(self.datasets):
            ds_first = self.datasets[idx]
            bounds = get_auto_bounds(ds_first)

            if bounds is None:
                print(f"Skipping timestep {idx}: Entirely unilluminated (Nightside).")
                idx += 1
                continue

            res_prev = final_median(ds_first, bounds['lat_min'], bounds['lat_max'],
                                              bounds['lon_min'], bounds['lon_max'])
            if res_prev is not None:
                df_prev = calculate_feature_metrics(res_prev)
                if df_prev.empty:
                    res_prev = None
            idx += 1

        if res_prev is None:
            print("Failed to initialize: No valid features found in any dataset.")
            return pd.DataFrame()

        print(f"Auto-Bounds Acquired: Lat [{bounds['lat_min']:.1f} to {bounds['lat_max']:.1f}] | "
              f"Lon [{bounds['lon_min']:.1f} to {bounds['lon_max']:.1f}]")

        ds_prev = self.datasets[idx-1]
        global_id_counter = int(df_prev['feature_id'].max()) + 1 if not df_prev.empty else 1

        for i in range(idx, len(self.datasets)):
            ds_curr = self.datasets[i]

            # Recalculate bounds dynamically to handle spacecraft/planetary rotation
            bounds_curr = get_auto_bounds(ds_curr)
            if bounds_curr is None:
                print(f"Skipping timestep {i}: Unilluminated.")
                continue

            res_curr = final_median(ds_curr, bounds_curr['lat_min'], bounds_curr['lat_max'],
                                              bounds_curr['lon_min'], bounds_curr['lon_max'])

            if res_curr is None: continue
            df_curr = calculate_feature_metrics(res_curr)
            if df_curr.empty: continue

            t1_dt = pd.to_datetime(ds_prev.time.values)[0]
            t2_dt = pd.to_datetime(ds_curr.time.values)[0]
            dt_seconds = (t2_dt - t1_dt).total_seconds()

            tracking_results = execute_tracking_pipeline(res_prev, res_curr, df_prev, df_curr, dt_seconds)

            if not tracking_results.empty:
                df_evo = df_prev.merge(tracking_results, on='feature_id')
                df_final, df_curr, global_id_counter = get_evolution_deltas(df_evo, df_curr, global_id_counter)

                df_final['time_t1'] = t1_dt
                df_final['time_t2'] = t2_dt
                self.master_log.append(df_final)

                t_str = f"{t1_dt.strftime('%Y-%m-%d %H:%M')} -> {t2_dt.strftime('%H:%M')}"

                print(f"Processed {t_str} | Found: {len(df_curr)} | Tracked & Verified: {len(df_final)}")

                plot_corrected_velocimetry(res_prev, res_curr, df_final, t_str)

            ds_prev, res_prev, df_prev = ds_curr, res_curr, df_curr

        if self.master_log:
            final_df = pd.concat(self.master_log, ignore_index=True)
            final_df.to_csv(self.output_csv, index=False)
            print(f"Processing complete! Saved to {self.output_csv}")
            return final_df

        print("Processing complete, but no valid vectors were tracked.")
        return pd.DataFrame()