import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import rotate

def read_hci_data(filepath,
                  dataset_name='object_hr_stacked_05'):
    with h5py.File(name=filepath, mode='r') as f_src:
        data = np.array(f_src[dataset_name])
        angles = np.array(f_src['header_' + dataset_name]['PARANG'])
        return data, angles

def derotate_images(images, angles, verbose=True):
    derotated_images = []
    for i in tqdm(range(len(images)), disable=not verbose):
        derotated_images.append(rotate(images[i], -angles[i], reshape=False))
    return np.array(derotated_images)

