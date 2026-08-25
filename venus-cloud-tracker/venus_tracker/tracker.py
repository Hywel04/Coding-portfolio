#Imports
import numpy as np
from scipy.signal import correlate2d
from skimage.morphology import h_maxima

#Physical tracking module for Venus cloud motion tracking
class VenusPhysicalTracker:
    """
    Handles physical coordinate transformations, cross-correlation matching,
    and Relaxation Labeling using Deformation Consistency (Ikegawa & Horinouchi, 2016).
    """
    def __init__(self, deg_per_px, dt_seconds, d_threshold=0.05, alpha=0.5, iterations=15):
        self.deg_per_px = deg_per_px
        self.dt_seconds = dt_seconds
        self.d = d_threshold
        self.alpha = alpha
        self.iters = iterations
        self.max_zonal_ms = 150.0
        self.max_merid_ms = 40.0
        self.R_venus = 6051800.0

    def px_to_ms(self, px_shift, latitude, is_zonal=False):
        deg_shift = px_shift * self.deg_per_px
        meters_per_deg_lat = (np.pi * self.R_venus) / 180.0
        if is_zonal:
            meters_per_deg = meters_per_deg_lat * np.cos(np.radians(latitude))
        else:
            meters_per_deg = meters_per_deg_lat
        return (deg_shift * meters_per_deg) / self.dt_seconds

    def get_ccs_candidates(self, img_t1, img_t2, r_t1, c_t1, r_t2, c_t2, lat, window_size=31, search_area=81):
        r_t1, c_t1 = int(r_t1), int(c_t1)
        r_t2, c_t2 = int(r_t2), int(c_t2)
        half_w, half_s = window_size // 2, search_area // 2

        # Boundary checks for BOTH independent image grids
        if r_t1-half_w < 0 or c_t1-half_w < 0 or r_t1+half_w >= img_t1.shape[0] or c_t1+half_w >= img_t1.shape[1]:
            return []
        if r_t2-half_s < 0 or c_t2-half_s < 0 or r_t2+half_s >= img_t2.shape[0] or c_t2+half_s >= img_t2.shape[1]:
            return []

        template = img_t1[r_t1-half_w : r_t1+half_w+1, c_t1-half_w : c_t1+half_w+1]
        search_window = img_t2[r_t2-half_s : r_t2+half_s+1, c_t2-half_s : c_t2+half_s+1]

        # NaN Guard: Abort if the template or search window is mostly missing data
        if np.isnan(template).sum() > (template.size * 0.5) or np.isnan(search_window).sum() > (search_window.size * 0.5):
            return []

        with np.errstate(all='ignore'):
            t_norm = template - np.nanmean(template)
            s_norm = search_window - np.nanmean(search_window)
            t_norm = np.nan_to_num(t_norm, nan=0.0)
            s_norm = np.nan_to_num(s_norm, nan=0.0)

        ccs = correlate2d(s_norm, t_norm, mode='same')

        if np.nanmax(ccs) == np.nanmin(ccs): return []
        ccs = (ccs - np.nanmin(ccs)) / (np.nanmax(ccs) - np.nanmin(ccs) + 1e-8)

        local_max = h_maxima(ccs, h=self.d)
        peak_coords = np.argwhere(local_max)

        candidates = []
        for p in peak_coords:
            val = ccs[p[0], p[1]]
            if val > 0.5:
                px_u, px_v = p[1] - half_s, p[0] - half_s
                v_u_ms = self.px_to_ms(px_u, lat, is_zonal=True)
                v_v_ms = self.px_to_ms(px_v, lat, is_zonal=False)

                if abs(v_u_ms) <= self.max_zonal_ms and abs(v_v_ms) <= self.max_merid_ms:
                    candidates.append({
                        'vec': np.array([px_u * self.deg_per_px, px_v * self.deg_per_px]),
                        'score': val,
                        'v_ms': np.array([v_u_ms, v_v_ms])
                    })
        return sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]

    def run_relaxation(self, feature_candidates, feature_distances):
        p = {}
        for f_id, cands in feature_candidates.items():
            Ik = len(cands)
            p[f_id] = np.zeros(Ik + 1)
            if Ik > 0:
                p[f_id][0] = 0.5  # Baseline probability for No-Match
                for idx, cand in enumerate(cands):
                    p[f_id][idx + 1] = cand['score']
                p[f_id] /= np.sum(p[f_id])
            else:
                p[f_id][0] = 1.0

        neighbors_map = {k: [] for k in p}
        for (k, l), dist in feature_distances.items():
            if dist <= 15.0: # Neighborhood radius (~1,500km)
                neighbors_map[k].append((l, dist))
                neighbors_map[l].append((k, dist))

        for _ in range(self.iters):
            q = {k: np.zeros(len(p[k])) for k in p}
            max_idx = {l: np.argmax(p[l]) for l in p}

            for k in p:
                if not neighbors_map[k]:
                    continue # Isolated cloud: skip neighbor math

                for l, dist_kl in neighbors_map[k]:
                    for i in range(len(p[k])):
                        for j in range(len(p[l])):
                            if i == 0:
                                c_kl = 0.5 if j == max_idx[l] else 0.0
                            elif j == 0:
                                c_kl = 0.0
                            else:
                                vec_i = feature_candidates[k][i-1]['vec']
                                vec_j = feature_candidates[l][j-1]['vec']
                                diff_sq = np.sum((vec_i - vec_j)**2)
                                L_kl_sq = (self.alpha * dist_kl)**2
                                c_kl = np.exp(-np.log(2) * diff_sq / (L_kl_sq + 1e-6))

                            q[k][i] += c_kl * p[l][j]

            new_p = {}
            for k in p:
                if not neighbors_map[k]:
                    new_p[k] = p[k]
                    continue

                num = p[k] * q[k]
                den = np.sum(num)
                new_pk = num / den if den > 0 else np.zeros_like(p[k])

                if len(p[k]) > 1:
                    match_probs = p[k][1:]
                    second_greatest = np.sort(match_probs)[-2] if len(match_probs) >= 2 else 0.0
                    new_pk[0] = max(new_pk[0], second_greatest)

                new_p[k] = new_pk / np.sum(new_pk) if np.sum(new_pk) > 0 else new_pk
            p = new_p

        winners = {}
        for k in p:
            best_idx = np.argmax(p[k])
            winners[k] = None if best_idx == 0 else feature_candidates[k][best_idx - 1]

        return winners
