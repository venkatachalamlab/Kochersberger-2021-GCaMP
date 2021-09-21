""" Functions used to crop and center volumetric recordings """

import json
import shutil

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from array_writer import TimestampedArrayWriter
from utils import get_metadata, get_times, get_slice

def mip(vol, axis):
    return np.max(vol, axis=axis)

def blur(img, sigma):
    return cv2.GaussianBlur(img, (sigma, sigma), 0)

def get_tform(ref, mov, iterations=500, similarity=1e-8):
    s = ref.shape
    ref = cv2.resize(ref, (s[1]//5, s[0]//5))
    mov = cv2.resize(mov, (s[1]//5, s[0]//5))

    w = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  similarity)

    input_mask = np.uint8(ref > -1)
    
    try:
        (_, w[:2, :]) = cv2.findTransformECC(ref, mov, w[:2, :], cv2.MOTION_EUCLIDEAN,
                                             criteria, input_mask, 5)
    except:
        print("ECC error")
    w[0, 2] = w[0, 2] * s[1] / (s[1]//5)
    w[1, 2] = w[1, 2] * s[0] / (s[0]//5)
    return w

def apply_tform(vol, tform, c_list):
    for c in c_list:
        for z in range(vol.shape[1]):
            vol[c, z, ...] = cv2.warpAffine(vol[c, z, ...], tform[:2, :], (vol.shape[3], vol.shape[2]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return vol

def get_distance_matrix(path):
    @lru_cache(maxsize=1500)
    def get_thumbnail(t):
        v = get_slice(path, t)
        rfp = mip(v[0], 0)
        return resize(rfp, np.array(rfp.shape) // 10)
    D = get_all_pdists(get_thumbnail,
                       list(range(shape_t)),
                       dist_fn=dist_corrcoef)
    return D + np.transpose(D)

def get_closest_coord(old, new_list):
    distances = []
    for pairs in new_list:
        distances.append((pairs[0] - old[0])^2 + (pairs[1] - old[1])^2)
    return new_list[np.argmin(np.array(distances))]

def find_center(v):
    cx, cy = [v.shape[3] // 2, v.shape[2] // 2]
    
    blurred = blur(mip(v[0], 0), 7)
    blurred[blurred<np.quantile(blurred, 0.997)] = 0
    blurred = blur(blurred, 131)
    blurred[blurred<np.quantile(blurred, 0.997)] = 0
    contours = measure.find_contours(blurred, np.quantile(blurred, 0.99))
    if len(contours)==1:
        cx, cy = int(np.mean(contours[0][:, 1])), int(np.mean(contours[0][:, 0]))
    elif len(contours)>1:
        candidates = []
        for contour in contours:
            candidates.append([int(np.mean(contour[:, 1])), int(np.mean(contour[:, 0]))])
        cx, cy = get_closest_coord((cx, cy), candidates)
    return cx, cy

def move_center(v):
    cx, cy = find_center(v)
    translation = np.eye(3, 3, dtype=np.float32)
    translation[0, 2] = cx - v.shape[3] // 2
    translation[1, 2] = cy - v.shape[2] // 2
    return apply_tform(v, translation, (0, 1))
  
  def center(src):
    src = Path(src)
    dst = src.parent / "centered"
    
    if not os.path.exists(dst):
        os.makedirs(dst)
    if os.path.isfile(dst / 'data.h5'):
        os.remove(dst / 'data.h5')


    metadata = get_metadata(src)
    shape_x = metadata['shape_x']
    shape_y = metadata['shape_y']
    shape_z = metadata['shape_z']
    shape_t = metadata['shape_t']
    shape_c = metadata['shape_c']
    new_shape_x = 300
    new_shape_y = 300
    times = get_times(src)
    shape = (shape_c, shape_z, new_shape_y, new_shape_x)
    writer = TimestampedArrayWriter(None, dst / 'data.h5', shape, dtype=np.uint8,
                                    groupname=None, compression="gzip", compression_opts=5)
    for t in tqdm(range(shape_t)):
        vol = get_slice(src, t)
        move_center(vol)
        vol = vol[:, :, 105:405, 105:405]
        vol = apply_lut(vol, 0.0, 255.0, newtype=np.uint8)
        writer.append_data((times[t], vol))
    writer.close()
    metadata['shape_x'] = 300
    metadata['shape_y'] = 300
    metadata['dtype'] = 'uint8'
    with open(dst / 'metadata.json', 'w') as outfile:
        json.dump(metadata, outfile, indent=4)
    shutil.copy(src / 'log.txt', dst)
