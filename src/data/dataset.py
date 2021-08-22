import settings
import data.handler as data_handler

import torch
import itertools
import numpy as np


def sigmoid(z):
    return (1 / (np.exp(-z))) - 1


def calc_weigths(fmri_data, min_weight=0.0, max_weight=2.0):
    weight = sigmoid(np.abs(fmri_data))  # calc 0 based sigmoid
    weight = np.clip(weight, min_weight, max_weight)  # clip values to desired range
    return weight


def generate_combinations(fmri_data):
    # add mean of all combinations
    mean_over_rep = np.expand_dims(fmri_data.mean(1), 1)
    fmri_data = np.append(fmri_data, mean_over_rep, axis=1)
    # add all posible combinations of repetitions
    rep_idx = range(settings.REPETITIONS)
    for comb_idx in itertools.combinations(rep_idx, 2):
        rep_comb = (fmri_data[:, comb_idx[0], :] + fmri_data[:, comb_idx[1], :]) / 2
        rep_comb = np.expand_dims(rep_comb, 1)
        fmri_data = np.append(fmri_data, rep_comb, axis=1)

    return fmri_data


class VidFMRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vid_files,
        fmri_data,
        transform=None,
        fmri_transform=None,
        fmri_noise=None,
        vid_rand_skip_cut=0.1,
    ):
        # prepare data
        self.prepare_data(vid_files, fmri_data, fmri_transform)
        # weights based on voxel activity
        self.weight = calc_weigths(self.fmri_data)
        # input and target transformations
        self.vid_rand_skip_cut = vid_rand_skip_cut
        self.fmri_noise = fmri_noise
        self.transform = transform
        self.fmri_transform = fmri_transform

    def prepare_data(self, vid_files, fmri_data, fmri_transform):
        # mean fmri value over repetitions
        if fmri_transform == "mean_over_rep":
            self.vid_files = vid_files
            self.fmri_data = fmri_data.mean(1)
        # create new fmri labels by combinations of repetitions
        elif fmri_transform == "combination_augment":
            # add all combinations of repetitions
            fmri_data = generate_combinations(fmri_data.copy())
            # flatten data
            N, R, V = np.shape(fmri_data)  # number of videos, repetitions, voxels
            self.vid_files = np.repeat(vid_files, R)  # repeat for each repetition
            self.fmri_data = fmri_data.reshape(N * R, V)
        # just flatten given data
        else:
            N, R, V = np.shape(fmri_data)  # number of videos, repetitions, voxels
            self.vid_files = np.repeat(vid_files, settings.REPETITIONS)
            self.fmri_data = fmri_data.reshape(N * R, V)

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        vid_fn = self.vid_files[idx]
        weight = self.weight[idx]
        # load video data
        vid_data, fps = data_handler.load_video(
            settings.VID_FOLDER + vid_fn,
            target_fps=settings.TARGET_FPS,
            random_skip_rate=self.vid_rand_skip_cut,
            random_cut_rate=self.vid_rand_skip_cut,
        )
        t, x, y, c = np.shape(vid_data)  # frames, x, y, channels
        # transform
        if self.transform:
            # reshape frames to channels: x, y, t*c
            vid_data = vid_data.transpose(1, 2, 0, 3).reshape(x, y, -1)
            # apply transformations accross channels
            transformed = self.transform(image=vid_data)
            # reshape back to: frames, channels, x, y
            vid_data = (
                transformed["image"]
                .reshape(settings.IMG_SIZE, settings.IMG_SIZE, t, c)
                .transpose(2, 3, 0, 1)
            )

        # label noise
        if self.fmri_noise is not None:
            noise_ratio = (1 - self.fmri_noise / 2) + (
                np.random.random(np.shape(fmri)) * self.fmri_noise
            )
            fmri = fmri.copy() * noise_ratio
            weight = weight.copy() * noise_ratio

        # pad or cut
        if t < settings.MAX_SEQ_LEN:
            padded = np.zeros(
                (settings.MAX_SEQ_LEN, c, settings.IMG_SIZE, settings.IMG_SIZE)
            )
            padded[-t:, :, :, :] = vid_data
            vid_data = padded
        # just in case sometimes rounding in load_video function can cause issues
        elif t > settings.MAX_SEQ_LEN:
            vid_data = vid_data[-settings.MAX_SEQ_LEN :, :, :, :]

        return {
            "vid_data": vid_data.astype(np.float32),
            "fmri": fmri.astype(np.float32),
            "frames": t,
            "weight": weight.astype(np.float32),
            "fps": fps.astype(np.float32),
        }


class VidDataset(torch.utils.data.Dataset):
    def __init__(self, vid_files, transform=None):
        self.vid_files = vid_files
        self.transform = transform

    def __len__(self):
        return len(self.vid_files)

    def __getitem__(self, idx):
        vid_fn = self.vid_files[idx]
        # load data
        vid_data, fps = data_handler.load_video(
            settings.VID_FOLDER + vid_fn, target_fps=settings.TARGET_FPS
        )
        t, x, y, c = np.shape(vid_data)  # frames, x, y, channels
        # transform
        if self.transform:
            # reshape frames to channels: x, y, t*c
            vid_data = vid_data.transpose(1, 2, 0, 3).reshape(x, y, -1)
            # apply transformations accross channels
            transformed = self.transform(image=vid_data)
            # reshape back to: frames, channels, x, y
            vid_data = (
                transformed["image"]
                .reshape(settings.IMG_SIZE, settings.IMG_SIZE, t, c)
                .transpose(2, 3, 0, 1)
            )
        # pad or cut
        if t < settings.MAX_SEQ_LEN:
            padded = np.zeros(
                (settings.MAX_SEQ_LEN, c, settings.IMG_SIZE, settings.IMG_SIZE)
            )
            padded[-t:, :, :, :] = vid_data
            vid_data = padded
        # just in case sometimes rounding in load_video function can cause issues
        elif t > settings.MAX_SEQ_LEN:
            vid_data = vid_data[-settings.MAX_SEQ_LEN :, :, :, :]

        return {
            "vid_data": vid_data.astype(np.float32),
            "frames": t,
            "fps": fps.astype(np.float32),
        }
