import os
import pickle
import numpy as np
import nibabel as nib

from decord import VideoReader
from decord import cpu
from decord.bridge import set_bridge

# Some funtions taken from Algonauts2021 starter notebook: 
# https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit

def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        ret_di = u.load()
    return ret_di


def saveasnii(brain_mask, nii_save_path, nii_data):
    img = nib.load(brain_mask)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)


def load_video(fn, target_fps=15, random_skip_rate=0.0, random_cut_rate=0.0):
    with open(fn, "rb") as f:
        os.set_blocking(f.fileno(), False)  # non blocking read, requires python 3.5+
        vr = VideoReader(f, ctx=cpu(0))

    vid_len = len(vr)
    fps = np.round(vr.get_avg_fps())

    sampling_rate = fps / target_fps
    frame_count = vid_len / sampling_rate

    max_skip = int(frame_count * random_skip_rate)
    max_cut = int(frame_count * random_cut_rate)
    start = 0
    cut = 0

    if max_skip > 0:
        start = np.random.randint(0, max_skip)
    if max_cut > 0:
        cut = np.random.randint(0, max_cut)

    frame_batch = np.round(np.arange(start, vid_len - cut, sampling_rate)).astype(int)
    if frame_batch[-1] >= vid_len:
        frame_batch[-1] = vid_len - 1

    data = vr.get_batch(frame_batch)
    data = data.asnumpy()
    return data, fps


def get_fmri(fmri_dir, ROI):
    """This function loads fMRI data into a numpy array for to a given ROI.
    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    ROI : str
        name of ROI.
    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """

    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)
    ROI_data_train = ROI_data["train"]

    if ROI == "WB":
        voxel_mask = ROI_data["voxel_mask"]
        return ROI_data_train, voxel_mask

    return ROI_data_train, None
