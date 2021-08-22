import settings
import data.handler as data_handler

import os
import numpy as np

import albumentations as A
import albumentations.pytorch as APT
import cv2

# TRANSFORMATIONS
def get_train_transform():
    return A.Compose(
        [
            A.Resize(settings.IMG_SIZE, settings.IMG_SIZE),
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.IAAPerspective(p=0.25),
            A.CoarseDropout(p=0.5),
            A.RandomBrightness(p=0.25),
            A.ToFloat(),
        ]
    )


def get_test_transform():
    return A.Compose([A.Resize(settings.IMG_SIZE, settings.IMG_SIZE), A.ToFloat(),])


# GET CHALLANGE FMRI DATA
def get_fmri_data_for_sub(sub, ROIs=settings.ROIs):
    fmri_ROI_mapping = {}
    fmri_voxel_total = 0

    # get selected voxel count for each ROI and create data mapping
    for ROI in ROIs:
        ROI_data, _ = data_handler.get_fmri(settings.FMRI_FOLDER + sub, ROI)
        ROI_voxel_count = ROI_data.shape[2]
        fmri_ROI_mapping[ROI] = [
            fmri_voxel_total,
            fmri_voxel_total + ROI_voxel_count,
            ROI_voxel_count,
        ]
        fmri_voxel_total += ROI_voxel_count

    # load fmri data
    sub_fmri_data = np.empty((settings.train_data_len, 3, fmri_voxel_total))
    for ROI in ROIs:
        ROI_mapping = fmri_ROI_mapping[ROI]
        ROI_data, _ = data_handler.get_fmri(settings.FMRI_FOLDER + sub, ROI)
        sub_fmri_data[:, :, ROI_mapping[0] : ROI_mapping[1]] = ROI_data

    return sub_fmri_data, fmri_ROI_mapping


def get_fmri_data(subs=settings.subs, ROIs=settings.ROIs):
    fmri_data = {}
    for sub in subs:
        sub_fmri_data, fmri_ROI_mapping = get_fmri_data_for_sub(sub, ROIs)
        fmri_data[sub] = {"mapping": fmri_ROI_mapping, "data": sub_fmri_data}

    return fmri_data


def get_data(subs=settings.subs, ROIs=settings.ROIs):
    vid_files = sorted(os.listdir(settings.VID_FOLDER))
    train_vid_files = vid_files[: settings.train_data_len]
    test_vid_files = vid_files[settings.train_data_len :]
    fmri_data = get_fmri_data(subs, ROIs)
    return fmri_data, train_vid_files, test_vid_files
