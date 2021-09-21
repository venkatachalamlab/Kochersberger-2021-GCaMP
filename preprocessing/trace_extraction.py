from math import floor
from pathlib import Path

import numpy as np
from tqdm import tqdm
import sklearn.neighbors as sk

from annotation.annotations_io import load_annotations
from utils.utils import get_metadata, get_slice




def _idx_from_coord(coord, shape):
    return max(floor(coord*shape - 1e-6), 0)


def fill_mask(mask, zyx, grid_radii, value):
    """Sets all elements about zyx indices in an area specified by
    grid_radii to the value. """

    copied_mask = np.copy(mask)

    z_idx = np.s_[int(max(0, zyx[0] - grid_radii[0])):
                  int(min(mask.shape[1] - 1, zyx[0] + grid_radii[0] + 1))]

    y_idx = np.s_[int(max(0, zyx[1] - grid_radii[1])):
                  int(min(mask.shape[2] - 1, zyx[1] + grid_radii[1] + 1))]

    x_idx = np.s_[int(max(0, zyx[2] - grid_radii[2])):
                  int(min(mask.shape[3] - 1, zyx[2] + grid_radii[2] + 1))]

    copied_mask[:, z_idx, y_idx, x_idx] = value

    return copied_mask


def extract_traces(data_path, kn_max=3,
                   grid_radii=(1, 4, 4), pixels_to_keep=40):
    """Uses coordinates.h5 file and creates an annotation datatable.
    Uses the datatable to extract traces and saves them in a numpy array."""

    p = Path(data_path)
    A, W = load_annotations(p)
    metadata = get_metadata(p)
    
    n_tracks = W.df.shape[0]
    annotated_times = np.unique(A.df["t_idx"])

    traces = np.zeros((metadata["shape_c"], n_tracks, metadata["shape_t"]))
    traces[:] = np.NaN

    for t in tqdm(range(metadata["shape_t"])):
        if t in annotated_times:
            V = get_slice(p, t)

            A_t = A.df[A.df['t_idx'] == t]
            tracks_t = np.unique(A_t["worldline_id"])
            n_tracks_t = tracks_t.shape[0]

            zyx = np.zeros((n_tracks_t, 3))

            for i, track in enumerate(tracks_t):
                A_t_n = A_t[A_t['worldline_id'] == track]
                zyx[i] = [_idx_from_coord(A_t_n['z'], V.shape[1]),
                          _idx_from_coord(A_t_n['y'], V.shape[2]),
                          _idx_from_coord(A_t_n['x'], V.shape[3])]

            tree = sk.KDTree(zyx)
            n_neighbors = min(n_tracks_t - 1, kn_max)
            neighbors = np.empty((n_tracks_t, n_neighbors))

            for i, coords in enumerate(zyx):
                neighbors[i] = tree.query(np.array([coords]),
                                          k=n_neighbors+1, return_distance=False)[0, 1:]

            for i, track in enumerate(tracks_t):
                mask = np.zeros_like(V)
                mask = fill_mask(mask, zyx[i], grid_radii, 1.0)

                for neighbor in neighbors[i]:
                    mask = fill_mask(mask, zyx[int(neighbor)], grid_radii, 0.0)

                masked_v = V * mask

                for c in range(metadata["shape_c"]):
                    non_zero_v = masked_v[c][np.nonzero(masked_v[c])]
                    if non_zero_v.shape[0] != 0:
                        traces[c, track, t] = np.mean(
                            np.sort(non_zero_v)[-pixels_to_keep: ])
            
    file_name = p / "traces.npy"
    np.save(file_name, traces)