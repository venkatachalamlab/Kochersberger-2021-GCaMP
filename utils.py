import json
from pathlib import Path

import h5py


def get_metadata(dataset_path: Path):
    json_filename = dataset_path / "metadata.json"
    with open(json_filename) as json_file:
        metadata = json.load(json_file)
    return metadata
  
def get_times(dataset_path: Path):
    h5_filename = dataset_path / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return f["times"][:]
  
def get_slice(dataset_path: Path, t):
    h5_filename = dataset_path / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return f["data"][t]

def apply_lut(x: np.ndarray, lo: float, hi: float, newtype=None) -> np.ndarray:
    if newtype is None:
        newtype = x.dtype
    y_float = (x-lo)/(hi-lo)
    y_clipped = np.clip(y_float, 0, 1)
    if np.issubdtype(newtype, np.integer):
        maxval = np.iinfo(newtype).max
    else:
        maxval = 1.0
    return (maxval*y_clipped).astype(newtype)
  
def mip_threeview(vol: np.ndarray, scale=(4,1,1)) -> np.ndarray:
    S = vol.shape[:3] * np.array(scale)
    output_shape = (S[1] + S[0],
                    S[2] + S[0])
    if vol.ndim == 4:
        output_shape = (*output_shape, 3)
    vol = np.repeat(vol, scale[0], axis=0)
    vol = np.repeat(vol, scale[1], axis=1)
    vol = np.repeat(vol, scale[2], axis=2)
    x = mip_x(vol)
    y = mip_y(vol)
    z = mip_z(vol)
    output = np.zeros(output_shape, dtype=vol.dtype)
    output[:S[1], :S[2]] = z
    output[:S[1], S[2]:] = x
    output[S[1]:, :S[2]] = y

    return output
