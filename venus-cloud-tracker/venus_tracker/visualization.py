#Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.patheffects as PathEffects

# Visualization functions for Venus cloud motion tracking
def plot_corrected_velocimetry(results_t1, results_t2, df_final, title_timestamp,
                               save_dir="data/processed/saved_pngs"):
    """Build a tracked-vector visualization. Saves a PNG to save_dir if given,
    otherwise returns the figure (e.g. for inline notebook display) without
    writing anything to disk."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    lon_t1, lat_t1 = results_t1['longitude'], results_t1['latitude']
    lon_t2, lat_t2 = results_t2['longitude'], results_t2['latitude']

    extent_t2 = [lon_t2.min(), lon_t2.max(), lat_t2.min(), lat_t2.max()]

    ax.imshow(results_t2['corrected_rad'], cmap='gray', extent=extent_t2, origin='lower', alpha=0.8)

    # T1 and T2 Boundaries
    ax.contour(lon_t1, lat_t1, ~np.isnan(results_t1['features']), levels=[0.5], colors='yellow', linestyles='dashed', linewidths=1.2)
    ax.contour(lon_t2, lat_t2, ~np.isnan(results_t2['features']), levels=[0.5], colors='cyan', linewidths=1.2)

    if not df_final.empty:
        u_vals, v_vals = pd.to_numeric(df_final['u'], errors='coerce'), pd.to_numeric(df_final['v'], errors='coerce')
        lon_vals, lat_vals = pd.to_numeric(df_final['centroid_lon'], errors='coerce'), pd.to_numeric(df_final['centroid_lat'], errors='coerce')
        valid_rows = u_vals.notna() & v_vals.notna() & lon_vals.notna() & lat_vals.notna()

        # Distinct Red Arrows, scaled 5x larger for visibility
        ax.quiver(lon_vals[valid_rows], lat_vals[valid_rows], u_vals[valid_rows], v_vals[valid_rows],
                  color='red', scale=0.2, scale_units='xy', angles='xy', width=0.005, headwidth=4)

        for _, row in df_final[valid_rows].iterrows():
            #Pure white text, pushed further from the center, with a dark background
            text = f"ID {int(row['feature_id'])} ({float(row['u_ms']):.0f} m/s)"

            ax.text(row['centroid_lon'] + 1.5, row['centroid_lat'] + 1.5, text,
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=0.5))

    ax.set_title(f"Venus Velocimetry | {title_timestamp}", fontsize=14)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    # Legend
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=1.2, ls='--', label='T1 Boundary'),
        Line2D([0], [0], color='cyan', lw=1.2, label='T2 Boundary'),
        Line2D([0], [0], color='red', marker='>', linestyle='none', markersize=8, label='Vector (5x Visual Scale)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.grid(True, linestyle=':', color='gray', alpha=0.3)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"velocity_map_{title_timestamp.replace(':', '').replace(' ', '_').replace('->', 'to')}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return None

    return fig